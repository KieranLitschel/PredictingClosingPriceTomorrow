import threading
import subprocess
from subprocess import PIPE
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import time


class ToDo:
    def __init__(self, model_constructor, cv, jobs, trainX, trainY, threads, total_memory=0.8, seed=0):
        self.model_constructor = model_constructor
        self.cv = cv
        self.trainX = trainX
        self.trainY = trainY
        self.folds = []
        kf = KFold(n_splits=cv, random_state=seed)
        for train, test in kf.split(trainX):
            self.folds.append((train, test))
        self.jobs = []
        for job in jobs:
            for i in range(0, cv):
                self.jobs.append((job, i))
        self.jobs.reverse()
        self.doing = [None] * threads
        self.total_memory = total_memory
        self.threads = threads
        self.memory_frac = total_memory / threads
        self.accuracies = {}
        for (job, fold) in self.jobs:
            self.accuracies[str(job)] = []
        self.results = {}

    def setNextJob(self, thread_number):
        if len(self.jobs) > 0:
            job, fold = self.jobs.pop()
            self.doing[thread_number] = (job, fold)
        else:
            self.doing[thread_number] = None
            done = True
            for job in self.doing:
                if job is not None:
                    done = False
                    break
            if done:
                print("Finishing up last fold, once fold completed message is recieved, type \"quit\" to exit")
        return self.doing[thread_number]

    def getJob(self, thread_number):
        return self.doing[thread_number]

    def doneJob(self, thread_number, accuracy):
        job, _ = self.getJob(thread_number)
        self.accuracies[str(job)].append(accuracy)
        if len(self.accuracies[str(job)]) == self.cv:
            accs = np.array(self.accuracies[str(job)])
            mean = np.average(accs)
            std = np.std(accs)
            self.results[str(job)] = {'mean': mean, 'std': std, 'accs': accs}
            print("Got mean of " + ("%.3f" % mean) + " and std of " + ("%.3f" % std) + " with parameters " + str(job))
        self.doing[thread_number] = None

    def failedJob(self, thread_number):
        job = self.doing[thread_number]
        self.jobs.append(job)
        self.doing.remove(job)

    def queueJobsDoing(self):
        for job_fold in self.doing:
            if job_fold is not None:
                self.jobs.append(job_fold)
        for i in range(0, len(self.doing)):
            self.doing[i] = None

    def setNumberOfThreads(self, threads=None, total_memory=None):
        self.queueJobsDoing()
        if total_memory is not None:
            self.total_memory = total_memory
        if threads is not None:
            self.threads = threads
        self.doing = [None] * self.threads
        self.threads = self.threads
        self.memory_frac = self.total_memory / threads

    def getTrainTest(self, fold):
        trainX = self.trainX[self.folds[fold][0]]
        trainY = self.trainY[self.folds[fold][0]]
        testX = self.trainX[self.folds[fold][1]]
        testY = self.trainY[self.folds[fold][1]]
        return trainX, trainY, testX, testY


class WorkerThread(threading.Thread):
    def __init__(self, thread_number, picklePath, pythonPath):
        threading.Thread.__init__(self)
        self.thread_number = thread_number
        self.picklePath = picklePath
        self.pythonPath = pythonPath

    def failed_job(self):
        writePickleLock.acquire()
        with open(self.picklePath, 'rb') as handle:
            toDo = pickle.load(handle)
        toDo.failedJob(self.thread_number)
        with open(self.picklePath, 'wb') as handle:
            pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        writePickleLock.release()

    def run(self):
        writePickleLock.acquire()
        with open(self.picklePath, 'rb') as handle:
            toDo = pickle.load(handle)
        nextJob = toDo.setNextJob(self.thread_number)
        with open(self.picklePath, 'wb') as handle:
            pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del toDo
        writePickleLock.release()
        more_jobs = True
        while more_jobs and kill_flag is False:
            changeProcsLock.acquire()
            if kill_flag:
                self.failed_job()
                changeProcsLock.release()
                break
            print("Starting fold " + str(nextJob[1] + 1) + " of job " + str(nextJob[0]))
            start = time.time()
            proc = subprocess.Popen(
                [self.pythonPath, "TensorflowSearchWorker.py", self.picklePath, str(self.thread_number)],
                stdout=PIPE)
            procs[self.thread_number] = proc
            changeProcsLock.release()
            output = proc.communicate()[0].decode("utf-8")
            if kill_flag:
                self.failed_job()
                break
            acc = float(output)
            writePickleLock.acquire()
            with open(self.picklePath, 'rb') as handle:
                toDo = pickle.load(handle)
            toDo.doneJob(self.thread_number, acc)
            oldJob = nextJob
            nextJob = toDo.setNextJob(self.thread_number)
            if nextJob is None:
                more_jobs = False
            with open(self.picklePath, 'wb') as handle:
                pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("  Finished fold " + str(oldJob[1] + 1) + " of job " + str(oldJob[0]) + (
                    " it took %.1f minutes" % ((time.time() - start) / 60)))
            writePickleLock.release()


class MultiThreadTensorGPUCrossValidation:
    def __init__(self, path="MTTGCV.pickle", pythonPath="venv\Scripts\python.exe", reload=False):
        global writePickleLock
        writePickleLock = threading.Lock()
        global changeProcsLock
        changeProcsLock = threading.Lock()
        self.picklePath = path
        self.pythonPath = pythonPath
        self.file_found = False
        if reload:
            try:
                with open(self.picklePath, 'rb') as handle:
                    toDo = pickle.load(handle)
                toDo.queueJobsDoing()
                self.thread_count = toDo.threads
                with open(self.picklePath, 'wb') as handle:
                    pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.file_found = True
            except FileNotFoundError:
                print("Error: Could not find the file at " + path)

    def create_new(self, trainX=None, trainY=None, model_constructor=None, search_type=None, param_grid=None,
                   iterations=None, cv=4, threads=4, total_memory=0.8, seed=0):
        if search_type == 'custom':
            jobs = param_grid
        elif search_type == 'grid':
            jobs = list(ParameterGrid(param_grid))
        elif search_type == 'random':
            jobs = list(ParameterSampler(param_grid, iterations, seed))
        toDo = ToDo(model_constructor, cv, jobs, trainX, trainY, threads, total_memory, seed)
        with open(self.picklePath, 'wb') as handle:
            pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.thread_count = threads
        self.file_found = True

    def change_threads_memory(self, threads=None, total_memory=None):
        if self.file_found is True:
            if total_memory is not None or threads is not None:
                with open(self.picklePath, 'rb') as handle:
                    toDo = pickle.load(handle)
                toDo.setNumberOfThreads(threads, total_memory)
                with open(self.picklePath, 'wb') as handle:
                    pickle.dump(toDo, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if threads is not None:
                self.thread_count = threads
        else:
            print("Error: You must create or load a search before doing this")

    def start(self):
        if self.file_found is True:
            global procs
            procs = [None] * self.thread_count
            threads = []
            global kill_flag
            kill_flag = False
            try:
                for thread_no in range(0, self.thread_count):
                    thread = WorkerThread(thread_no, self.picklePath, self.pythonPath)
                    thread.start()
                    threads.append(thread)
                msg = ""
                while msg != "quit":
                    print("Type quit at any time to end the search.")
                    msg = input()
            finally:
                print("Making sure all processes are killed, please be patient")
                changeProcsLock.acquire()
                kill_flag = True
                for proc in procs:
                    if proc is not None:
                        proc.kill()
                changeProcsLock.release()
                for thread in threads:
                    thread.join()
                print("All processes safely killed")
        else:
            print("Error: You must create or load a search before doing this")

    def getResults(self):
        if self.file_found is True:
            with open(self.picklePath, 'rb') as handle:
                toDo = pickle.load(handle)
            return toDo.results
        else:
            print("Error: You must create or load a search before doing this")
