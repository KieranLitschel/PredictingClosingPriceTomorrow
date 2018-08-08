CREATE TABLE fundamentals
(
  permno         INT        NOT NULL,
  public_date    DATE       NOT NULL,
  ticker         VARCHAR(5) NULL,
  Accrual        FLOAT      NULL,
  adv_sale       FLOAT      NULL,
  aftret_eq      FLOAT      NULL,
  aftret_equity  FLOAT      NULL,
  aftret_invcapx FLOAT      NULL,
  at_turn        FLOAT      NULL,
  bm             FLOAT      NULL,
  capei          FLOAT      NULL,
  capital_ratio  FLOAT      NULL,
  cash_debt      FLOAT      NULL,
  cash_lt        FLOAT      NULL,
  cash_ratio     FLOAT      NULL,
  cfm            FLOAT      NULL,
  curr_debt      FLOAT      NULL,
  curr_ratio     FLOAT      NULL,
  de_ratio       FLOAT      NULL,
  debt_assets    FLOAT      NULL,
  debt_at        FLOAT      NULL,
  debt_capital   FLOAT      NULL,
  debt_ebitda    FLOAT      NULL,
  debt_invcap    FLOAT      NULL,
  dltt_be        FLOAT      NULL,
  equity_invcap  FLOAT      NULL,
  evm            FLOAT      NULL,
  gpm            FLOAT      NULL,
  GProf          FLOAT      NULL,
  int_debt       FLOAT      NULL,
  int_totdebt    FLOAT      NULL,
  intcov_ratio   FLOAT      NULL,
  lt_debt        FLOAT      NULL,
  lt_ppent       FLOAT      NULL,
  npm            FLOAT      NULL,
  ocf_lct        FLOAT      NULL,
  opmad          FLOAT      NULL,
  opmbd          FLOAT      NULL,
  pcf            FLOAT      NULL,
  pe_exi         FLOAT      NULL,
  pe_inc         FLOAT      NULL,
  profit_lct     FLOAT      NULL,
  ps             FLOAT      NULL,
  ptb            FLOAT      NULL,
  ptpm           FLOAT      NULL,
  quick_ratio    FLOAT      NULL,
  RD_SALE        FLOAT      NULL,
  rect_turn      FLOAT      NULL,
  roa            FLOAT      NULL,
  roce           FLOAT      NULL,
  roe            FLOAT      NULL,
  sale_equity    FLOAT      NULL,
  sale_invcap    FLOAT      NULL,
  short_debt     FLOAT      NULL,
  staff_sale     FLOAT      NULL,
  totdebt_invcap FLOAT      NULL,
  CONSTRAINT fundamentals_permno_public_date_pk
  UNIQUE (permno, public_date)
)
  ENGINE = InnoDB;

CREATE INDEX fundamentals_ticker_public_date_index
  ON fundamentals (ticker, public_date);

CREATE TABLE tickers
(
  ticker                 VARCHAR(5)  NOT NULL,
  sector                 VARCHAR(30) NOT NULL,
  firstDay               DATE        NULL,
  lastUpdated            DATE        NULL,
  averageUpward          FLOAT       NULL,
  averageDownward        FLOAT       NULL,
  averageUpward_backup   FLOAT       NULL,
  averageDownward_backup FLOAT       NULL,
  PRIMARY KEY (ticker, sector)
)
  ENGINE = InnoDB;

ALTER TABLE fundamentals
  ADD CONSTRAINT fundamentals_tickers_ticker_fk
FOREIGN KEY (ticker) REFERENCES tickers (ticker);

CREATE TABLE timeseriesdaily
(
  ticker            VARCHAR(5) NOT NULL,
  date              DATE       NOT NULL,
  dateTmrw          DATE       NULL,
  open              FLOAT      NULL,
  high              FLOAT      NULL,
  low               FLOAT      NULL,
  close             FLOAT      NULL,
  adjClose          FLOAT      NULL,
  volume            BIGINT     NULL,
  lastFundamental   DATE       NULL,
  adjClosePChange   FLOAT      NULL,
  pDiffClose5SMA    FLOAT      NULL,
  pDiffClose8SMA    FLOAT      NULL,
  pDiffClose13SMA   FLOAT      NULL,
  rsi               FLOAT      NULL,
  pDiffCloseUpperBB FLOAT      NULL,
  pDiffCloseLowerBB FLOAT      NULL,
  pDiff20SMAAbsBB   FLOAT      NULL,
  pDiff5SMA8SMA     FLOAT      NULL,
  pDiff5SMA13SMA    FLOAT      NULL,
  pDiff8SMA13SMA    FLOAT      NULL,
  macdHist          FLOAT      NULL,
  deltaMacdHist     FLOAT      NULL,
  stochPK           FLOAT      NULL,
  stochPD           FLOAT      NULL,
  adx               FLOAT      NULL,
  pDiffPdiNdi       FLOAT      NULL,
  obvGrad5          FLOAT      NULL,
  obvGrad8          FLOAT      NULL,
  obvGrad13         FLOAT      NULL,
  adjCloseGrad5     FLOAT      NULL,
  adjCloseGrad8     FLOAT      NULL,
  adjCloseGrad13    FLOAT      NULL,
  adjCloseGrad20    FLOAT      NULL,
  adjCloseGrad35    FLOAT      NULL,
  adjCloseGrad50    FLOAT      NULL,
  `4_80_20`         INT        NULL,
  `2_80_20`         INT        NULL,
  `4_60_20_20`      INT        NULL,
  `4_60_20_20_wrds` INT        NULL,
  PRIMARY KEY (ticker, date),
  CONSTRAINT timeseriesdaily_tickers_ticker_fk
  FOREIGN KEY (ticker) REFERENCES tickers (ticker)
)
  ENGINE = InnoDB;

CREATE INDEX timeseriesdaily_ticker_date_index
  ON timeseriesdaily (ticker, date);

CREATE INDEX timeseriesdaily_ticker_lastFundamental_index
  ON timeseriesdaily (ticker, lastFundamental);


