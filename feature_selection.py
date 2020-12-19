import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from sklearn import tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler



#Data
data = yf.download("AAPL", "1995-1-1", "2019-12-31")
open = data["Open"]
high = data["High"]
low = data["Low"]
close = data["Adj Close"]
volume = data["Volume"]

#Features
dataset = pd.DataFrame(index=data.index)
dataset["Open"] = np.log(open / open.shift(1))
dataset["High"] = np.log(high / high.shift(1))
dataset["Low"] = np.log(low / low.shift(1))
dataset["Close"] = np.log(data["Close"] / data["Close"].shift(1))
dataset["Adj Close"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
dataset["upperband"], dataset["middleband"], dataset["lowerband"] = ta.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
dataset["real"] = ta.DEMA(close, timeperiod=30)
dataset["real2"] = ta.EMA(close, timeperiod=30)
dataset["real3"] = ta.HT_TRENDLINE(close)
dataset["real4"] = ta.KAMA(close, timeperiod=30)
dataset["real5"] = ta.MA(close, timeperiod=30, matype=0)
dataset["mama"], dataset["fama"] = ta.MAMA(close)
dataset["real6"] = ta.MAVP(close, dataset["real5"], minperiod=2, maxperiod=30, matype=0)
dataset["real7"] = ta.MIDPOINT(close, timeperiod=14)
dataset["real8"] = ta.MIDPRICE(high, low, timeperiod=14)
dataset["real9"] = ta.SAR(high, low, acceleration=0, maximum=0)
dataset["real10"] = ta.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
dataset["real11"] = ta.SMA(close, timeperiod=30)
dataset["real12"] = ta.T3(close, timeperiod=5, vfactor=0)
dataset["real13"] = ta.TEMA(close, timeperiod=30)
dataset["real14"] = ta.TRIMA(close, timeperiod=30)
dataset["real15"] = ta.WMA(close, timeperiod=30)
dataset["real16"] = ta.ADX(high, low, close, timeperiod=14)
dataset["real17"] = ta.ADXR(high, low, close, timeperiod=14)
dataset["real18"] = ta.APO(close, fastperiod=12, slowperiod=26, matype=0)
dataset["aroondown"], dataset["roonup"] = ta.AROON(high, low, timeperiod=14)
dataset["real19"] = ta.AROONOSC(high, low, timeperiod=14)
dataset["real20"] = ta.BOP(open, high, low, close)
dataset["real21"] = ta.CCI(high, low, close, timeperiod=14)
dataset["real22"] = ta.CMO(close, timeperiod=14)
dataset["real23"] = ta.DX(high, low, close, timeperiod=14)
dataset["macd"], dataset["macdsignal"], dataset["macdhist"] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
dataset["macd2"], dataset["macdsignal2"], dataset["macdhist2"] = ta.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
dataset["macd3"], dataset["macdsignal3"], dataset["macdhist3"] = ta.MACDFIX(close, signalperiod=9)
dataset["real24"] = ta.MFI(high, low, close, volume, timeperiod=14)
dataset["real25"] = ta.MINUS_DI(high, low, close, timeperiod=14)
dataset["real26"] = ta.MINUS_DM(high, low, timeperiod=14)
dataset["real27"] = ta.MOM(close, timeperiod=10)
dataset["real28"] = ta.PLUS_DI(high, low, close, timeperiod=14)
dataset["real29"] = ta.PLUS_DM(high, low, timeperiod=14)
dataset["real30"] = ta.PPO(close, fastperiod=12, slowperiod=26, matype=0)
dataset["real31"] = ta.ROC(close, timeperiod=10)
dataset["real32"] = ta.ROCP(close, timeperiod=10)
dataset["real33"] = ta.ROCR(close, timeperiod=10)
dataset["real34"] = ta.ROCR100(close, timeperiod=10)
dataset["real35"] = ta.RSI(close, timeperiod=14)
dataset["slowk"], dataset["slowd"] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
dataset["fastk"], dataset["fastd"] = ta.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
dataset["fastk2"], dataset["fastd2"] = ta.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
dataset["real36"] = ta.TRIX(close, timeperiod=30)
dataset["real37"] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
dataset["real38"] = ta.WILLR(high, low, close, timeperiod=14)
dataset["real39"] = ta.AD(high, low, close, volume)
dataset["real40"] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
dataset["real41"] = ta.OBV(close, volume)
dataset["real42"] = ta.ATR(high, low, close, timeperiod=14)
dataset["real43"] = ta.NATR(high, low, close, timeperiod=14)
dataset["real44"] = ta.TRANGE(high, low, close)
dataset["real45"] = ta.AVGPRICE(open, high, low, close)
dataset["real46"] = ta.MEDPRICE(high, low)
dataset["real47"] = ta.TYPPRICE(high, low, close)
dataset["real48"] = ta.WCLPRICE(high, low, close)
dataset["real49"] = ta.HT_DCPERIOD(close)
dataset["real50"] = ta.HT_DCPHASE(close)
dataset["inphase"], dataset["quadrature"] = ta.HT_PHASOR(close)
dataset["sine"], dataset["leadsine"] = ta.HT_SINE(close)
dataset["integer"] = ta.HT_TRENDMODE(close)
dataset["integer2"] = ta.CDL2CROWS(open, high, low, close)
dataset["integer4"] = ta.CDL3BLACKCROWS(open, high, low, close)
dataset["integer5"] = ta.CDL3INSIDE(open, high, low, close)
dataset["integer6"] = ta.CDL3LINESTRIKE(open, high, low, close)
dataset["integer7"] = ta.CDL3OUTSIDE(open, high, low, close)
dataset["integer8"] = ta.CDL3STARSINSOUTH(open, high, low, close)
dataset["integer9"] = ta.CDL3WHITESOLDIERS(open, high, low, close)
dataset["integer10"] = ta.CDLABANDONEDBABY(open, high, low, close, penetration=0)
dataset["integer11"] = ta.CDLADVANCEBLOCK(open, high, low, close)
dataset["integer12"] = ta.CDLBELTHOLD(open, high, low, close)
dataset["integer13"] = ta.CDLBREAKAWAY(open, high, low, close)
dataset["integer14"] = ta.CDLCLOSINGMARUBOZU(open, high, low, close)
dataset["integer15"] = ta.CDLCONCEALBABYSWALL(open, high, low, close)
dataset["integer16"] = ta.CDLCOUNTERATTACK(open, high, low, close)
dataset["integer17"] = ta.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
dataset["integer18"] = ta.CDLDOJI(open, high, low, close)
dataset["integer19"] = ta.CDLDOJISTAR(open, high, low, close)
dataset["integer20"] = ta.CDLDRAGONFLYDOJI(open, high, low, close)
dataset["integer21"] = ta.CDLENGULFING(open, high, low, close)
dataset["integer22"] = ta.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
dataset["integer23"] = ta.CDLEVENINGSTAR(open, high, low, close, penetration=0)
dataset["integer24"] = ta.CDLGAPSIDESIDEWHITE(open, high, low, close)
dataset["integer25"] = ta.CDLGRAVESTONEDOJI(open, high, low, close)
dataset["integer26"] = ta.CDLHAMMER(open, high, low, close)
dataset["integer27"] = ta.CDLHANGINGMAN(open, high, low, close)
dataset["integer28"] = ta.CDLHARAMI(open, high, low, close)
dataset["integer29"] = ta.CDLHARAMICROSS(open, high, low, close)
dataset["integer30"] = ta.CDLHIGHWAVE(open, high, low, close)
dataset["integer31"] = ta.CDLHIKKAKE(open, high, low, close)
dataset["integer32"] = ta.CDLHIKKAKEMOD(open, high, low, close)
dataset["integer33"] = ta.CDLHOMINGPIGEON(open, high, low, close)
dataset["integer34"] = ta.CDLIDENTICAL3CROWS(open, high, low, close)
dataset["integer35"] = ta.CDLINNECK(open, high, low, close)
dataset["integer36"] = ta.CDLINVERTEDHAMMER(open, high, low, close)
dataset["integer37"] = ta.CDLKICKING(open, high, low, close)
dataset["integer38"] = ta.CDLKICKINGBYLENGTH(open, high, low, close)
dataset["integer39"] = ta.CDLLADDERBOTTOM(open, high, low, close)
dataset["integer40"] = ta.CDLLONGLEGGEDDOJI(open, high, low, close)
dataset["integer41"] = ta.CDLLONGLINE(open, high, low, close)
dataset["integer42"] = ta.CDLMARUBOZU(open, high, low, close)
dataset["integer43"] = ta.CDLMATCHINGLOW(open, high, low, close)
dataset["integer44"] = ta.CDLMATHOLD(open, high, low, close, penetration=0)
dataset["integer45"] = ta.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
dataset["integer46"] = ta.CDLMORNINGSTAR(open, high, low, close, penetration=0)
dataset["integer47"] = ta.CDLONNECK(open, high, low, close)
dataset["integer48"] = ta.CDLPIERCING(open, high, low, close)
dataset["integer49"] = ta.CDLRICKSHAWMAN(open, high, low, close)
dataset["integer50"] = ta.CDLRISEFALL3METHODS(open, high, low, close)
dataset["integer51"] = ta.CDLSEPARATINGLINES(open, high, low, close)
dataset["integer52"] = ta.CDLSHOOTINGSTAR(open, high, low, close)
dataset["integer53"] = ta.CDLSHORTLINE(open, high, low, close)
dataset["integer54"] = ta.CDLSPINNINGTOP(open, high, low, close)
dataset["integer55"] = ta.CDLSTALLEDPATTERN(open, high, low, close)
dataset["integer56"] = ta.CDLSTICKSANDWICH(open, high, low, close)
dataset["integer57"] = ta.CDLTAKURI(open, high, low, close)
dataset["integer58"] = ta.CDLTASUKIGAP(open, high, low, close)
dataset["integer59"] = ta.CDLTHRUSTING(open, high, low, close)
dataset["integer60"] = ta.CDLTRISTAR(open, high, low, close)
dataset["integer61"] = ta.CDLUNIQUE3RIVER(open, high, low, close)
dataset["integer62"] = ta.CDLUPSIDEGAP2CROWS(open, high, low, close)
dataset["integer63"] = ta.CDLXSIDEGAP3METHODS(open, high, low, close)
dataset["real51"] = ta.BETA(high, low, timeperiod=5)
dataset["real52"] = ta.CORREL(high, low, timeperiod=30)
dataset["real53"] = ta.LINEARREG(close, timeperiod=14)
dataset["real54"] = ta.LINEARREG_ANGLE(close, timeperiod=14)
dataset["real55"] = ta.LINEARREG_INTERCEPT(close, timeperiod=14)
dataset["real56"] = ta.LINEARREG_SLOPE(close, timeperiod=14)
dataset["real57"] = ta.STDDEV(close, timeperiod=5, nbdev=1)
dataset["real58"] = ta.TSF(close, timeperiod=14)
dataset["real59"] = ta.VAR(close, timeperiod=5, nbdev=1)

#Dependent variable
dataset["target"] = np.where(close - close.shift() > 0, 1, -1)
dataset["target"] = dataset["target"].shift(-1)
dataset = dataset.dropna()

#Featur selection methods
def feature_selection(method, k, data):
    
    #chi-square (categorical features)
    if method == "chi_square":   
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        del(data["target"])
        X = MinMaxScaler().fit_transform(X)
        selector = SelectKBest(chi2, k=k)
        fitting = selector.fit_transform(X, y)
        selected = pd.DataFrame()
        selected["status"] = selector.get_support()
        selected["selected_features"] = data.drop(["target"], 1).columns
        selected = selected["selected_features"][selected["status"] == True]
        new_dataset = pd.DataFrame()
        for i in selected:
            new_dataset[i] = data[i]
        return new_dataset
    
    #ANOVA F-value (quantitative features)
    if method == "anova":
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif
        #Determining X, y
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        #del(data["target"])
        X = StandardScaler().fit_transform(X)
        selector = SelectKBest(f_classif, k=k)
        fitting = selector.fit_transform(X, y)
        selected = pd.DataFrame()
        selected["status"] = selector.get_support()
        selected["selected_features"] = data.drop(["target"], 1).columns
        selected = selected["selected_features"][selected["status"] == True]
        new_dataset = pd.DataFrame()
        for i in selected:
            new_dataset[i] = data[i]
        return new_dataset        
    
    #Forward Search
    if method == "random_forest":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import roc_auc_score 
        from mlxtend.feature_selection import SequentialFeatureSelector
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        del(data["target"])
        X = StandardScaler().fit_transform(X)
        selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                  k_features=k, forward=True, 
                                  verbose=2, scoring="roc_auc", cv=4)
        selector.fit(X, y)
        selected = pd.DataFrame()
        selected["no"] = selector.k_feature_idx_
        selected["selected_features"] = float()
        x = 0
        for i in selected["no"]:
            selected["selected_features"].iloc[x] = data.columns[i]
            x += 1
        new_dataset = pd.DataFrame()
        for i in selected["selected_features"]:
            new_dataset[i] = data[i]
        return new_dataset
    
    #Recursive Feature Elimination
    if method == "rfe":
        from sklearn.feature_selection import RFE
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        del(data["target"])
        X = StandardScaler().fit_transform(X)
        estimator = tree.DecisionTreeClassifier()
        selector = RFE(estimator, k, step=1)
        fitting = selector.fit(X, y)
        selected = pd.DataFrame()
        selected["status"] = fitting.get_support()
        selected["selected_features"] = data.drop(["target"], 1).columns
        selected = selected["selected_features"][selected["status"] == True]
        new_dataset = pd.DataFrame()
        for i in selected:
            new_dataset[i] = data[i]
        return new_dataset
    
    #Feature Importance with Extra Trees Classifier
    if method == "extra_tree":
        from sklearn.ensemble import ExtraTreesClassifier
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        del(data["target"])
        X = StandardScaler().fit_transform(X)
        model = ExtraTreesClassifier(n_estimators=k)
        model.fit(X, y)
        selected = pd.DataFrame()
        selected["status"] = model.feature_importances_
        selected["selected_features"] = data.columns
        selected = selected.sort_values(by=["status"])
        selected = selected[-k:]
        new_dataset = pd.DataFrame()
        for i in selected["selected_features"]:
            new_dataset[i] = data[i]
        return new_dataset
    
    #Principle Component Analysis
    if method == "pca":
        X = np.array(data.drop(["target"], 1))
        y = np.array(data["target"])
        del(data["target"])
        X = StandardScaler().fit_transform(X)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k)
        new_dataset = pca.fit_transform(X)
        return new_dataset


#selected_features = feature_selection("anova", 5, dataset)
