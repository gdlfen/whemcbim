# 第四轮接口探测


## A.新浪日线
len=145614 前 300=[{"day":"2021-05-28","open":"2240.970","high":"2250.000","low":"2205.510","close":"2230.000","volume":"2580731"},{"day":"2021-05-31","open":"2215.500","high":"2218.350","low":"2172.000","close":"2218.000","volume":"3173900"},{"day":"2021-06-01","open":"2222.000","high":"2248.950","low":"2206.000","c
条数=1300 首={'day': '2021-05-28', 'open': '2240.970', 'high': '2250.000', 'low': '2205.510', 'close': '2230.000', 'volume': '2580731'} 末={'day': '2026-10-09', 'open': '1255.420', 'high': '1282.000', 'low': '1255.420', 'close': '1263.000', 'volume': '3511051'}
[OK] A.新浪日线


## A2.腾讯日线
键: ['qfqday', 'qt', 'mx_price', 'prec', 'version']
qfqday 条数=640 首=['2024-02-19', '1603.531', '1561.091', '1603.591', '1553.661', '36977.000'] 末=['2026-10-09', '1255.420', '1263.000', '1282.000', '1255.420', '35111.000']
[OK] A2.腾讯日线


## B.行业板块清单
[push2.eastmoney.com] 第1次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2.eastmoney.com] 第2次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2.eastmoney.com] 第3次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2delay.eastmoney.com] 第1次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2delay.eastmoney.com] 第2次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[push2delay.eastmoney.com] 第3次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[OK] B.行业板块清单


## C.东财行业板块代码
行数=1 首行={"SECUCODE": "600519.SH", "SECURITY_CODE": "600519", "SECURITY_NAME_ABBR": "贵州茅台", "REPORTDATE": "2026-06-30 00:00:00", "PUBLISHNAME": "白酒Ⅱ", "BOARD_CODE": "BK1277", "BOARD_NAME": "白酒Ⅱ"}
[OK] C.东财行业板块代码


## D.涨跌家数字段
[1.000001] 第1次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[1.000001] 第2次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[1.000001] 第3次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[1.000001,0.399001,0.399006] 第1次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[1.000001,0.399001,0.399006] 第2次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[1.000001,0.399001,0.399006] 第3次失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[OK] D.涨跌家数字段


## E.新浪资金流
条数=120
首行={"opendate": "2026-10-09", "trade": "1263.4500", "changeratio": "0.00609975", "turnover": "27.7781", "netamount": "632979412.0300", "ratioamount": "0.143711", "r0": "3160151920.9800", "r1": "1150801891.3900", "r2": "88465222.0000", "r3": "0.0000", "r0_net": "643814361.2200", "r1_net": "-17008023.1900", "r2_net": "6173074.0000", "r3_net": "0.0000"}
末行={"opendate": "2026-04-13", "trade": "1445.5000", "changeratio": "-0.00581859", "turnover": "19.9181", "netamount": "-291301537.0800", "ratioamount": "-0.0811291", "r0": "2357925437.3900", "r1": "1121014273.7500", "r2": "111655980.9500", "r3": "0.0000", "r0_net": "-193678347.6700", "r1_net": "-83522822.1600", "r2_net": "-14100367.2500", "r3_net": "0.0000"}
键=['opendate', 'trade', 'changeratio', 'turnover', 'netamount', 'ratioamount', 'r0', 'r1', 'r2', 'r3', 'r0_net', 'r1_net', 'r2_net', 'r3_net']
创业板 300750 条数=5 首行={"opendate": "2026-10-09", "trade": "297.7600", "changeratio": "0.0383958", "turnover": "126.027", "netamount": "2063050419.7600", "ratioamount": "0.129818", "r0": "15421788564.4600", "r1": "688690059.1300", "r2": "4194530.0000", "r3": "29873.0000", "r0_net": "1958163442.7200", "r1_net": "104532917.0400", "r2_net": "354060.0000", "r3_net": "0.0000"}
[分钟/MoneyFlow.ssl_qsfx_zjlrqs?page] status=200 前 200=[{"opendate":"2026-10-09","trade":"1263.4500","changeratio":"0.00609975","turnover":"27.7781","netamount":"632979412.0300","ratioamount":"0.143711","r0_net":"643814361.2200","r0_ratio":"0.14617106","r
[分钟] 失败 ConnectionError(ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')))
[OK] E.新浪资金流


## F.东财F10行业
[RPT_F10_CORETHEME_BOARDTYPE] {"version":"6bd7a5547d761227cc8786cb61c807e0","result":{"pages":6,"data":[{"SECUCODE":"600519.SH","SECURITY_CODE":"600519","SECURITY_NAME_ABBR":"贵州茅台","BOARD_CODE":"999","BOARD_NAME":"茅指数","SELECTED_BOARD_REASON":null,"IS_PRECISE":"0","BOARD_RANK":11,"BOARD_YIELD":null,"NEW_BOARD_CODE":"BK0999","DERIVE_BOARD_CODE":"BI0999","BOARD_TYPE":null,"BOARD_LEVEL":null},{"SECUCODE":"600519.SH","SECURITY_COD
[RPT_F10_BASIC_ORGINFO] {"version":"9277664608da94d87ea60bd19f747740","result":{"pages":1,"data":[{"SECUCODE":"600519.SH","SECURITY_CODE":"600519","SECURITY_NAME_ABBR":"贵州茅台","ORG_CODE":"10002602","ORG_NAME":"贵州茅台酒股份有限公司","ORG_NAME_EN":"Kweichow Moutai Co.,Ltd.","FORMERNAME":"贵州茅台→G茅台","STR_CODEA":"600519","STR_NAMEA":"贵州茅台","STR_CODEB":null,"STR_NAMEB":null,"STR_CODEH":null,"STR_NAMEH":null,"SECURITY_TYPE":"上交所主板A股","EM
[OK] F.东财F10行业