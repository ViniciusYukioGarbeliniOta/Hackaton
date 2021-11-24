import pandas as pd
import numpy as np 
from yellowbrick.cluster import silhouette_visualizer
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler , PowerTransformer 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering,SpectralClustering
from sklearn.metrics import silhouette_samples , silhouette_score 
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly_express as px
import pylab
%matplotlib inline
%run '/home/vinicius_ota/Área de Trabalho/python_functions/connect_dw.py'


### Cria tabela que gera os clusters:
data = pd.read_sql_query("""SELECT 
                            nr_carteirinha ,
                            (max(dt_autorizacao::date) - min(dt_autorizacao::date)) AS tempo_entre_consultas,
                            current_date::date - max(dt_autorizacao::date) AS recencia,
                            count(DISTINCT nr_guia ) AS quantidade_guias,
                            sum(vl_unit_proced_guia*qt_proced_autorizado) AS sinistro_total
                        FROM  mv_data.invoices i 
                        WHERE status_guia = 'AUTORIZADA'
                        GROUP BY 1""", 
                         con=get_engine())
### Ajusta os clusters:
def preprocessing_feature( data , scaler_type  ):
    if scaler_type == 'StandarScaler': 
        std = StandardScaler().fit_transform( data )
        return pd.DataFrame(StandardScaler().fit_transform( data ) , columns = data.columns)
    if scaler_type == 'RobustScaler': 
        return pd.DataFrame(RobustScaler().fit_transform( data ) , columns = data.columns) 
    if scaler_type == 'MinMaxScaler': 
        return pd.DataFrame(MinMaxScaler().fit_transform( data ) , columns = data.columns)
    if scaler_type == 'Power': 
        return pd.DataFrame(PowerTransformer(method='box-cox').fit_transform(data) , columns = data.columns)
fitKmeans = KMeans(n_clusters=3).fit(preprocessing_feature(DataCluster,'RobustScaler'))
DataCluster['kmeans_cluster'] = pd.Series(fitKmeans.labels_, index=DataCluster.index)
DataCluster['kmeans_cluster'] = DataCluster['kmeans_cluster'].map({0:'A',1:'B',2:'C'})