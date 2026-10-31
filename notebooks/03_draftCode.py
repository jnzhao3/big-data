
import langchain
from langchain.tools import tool

from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransform
from pyspark.sql.functions import pandas_udf


import pandas as pd
import numpy as np


spark = SparkSession.builder \
    .appName("FrequentItems") \
    .master("yarn") \
    .config("spark.submit.deployMode", "client") \
    .config("spark.driver.memory", "16g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.instances", "10") \
    .config("spark.sql.shuffle.partitions","10")\
    .getOrCreate()

#idea is to store the 2 tables to be joined as csv in HDFS
#partitioning strategy: 


#this will be a tool for the LLM agent: data profiling
#it returns col information as a string Column:city|Type:str|Samples:Jersey City,Brooklyn 
#this is later converted as embeddings (column serialization)
@tool
def dataProfiling(filePath:str)->str:
    '''Generates a data profile for a given file. Outputs list of columns, data type, few sample values'''
    profile=[] #store profile information as a list of string
    df = spark.read.csv(
        filePath,
        sep="\t",
        inferSchema=True        
        )
    
    for col in df.columns:

        #select 5 samples of that column and add it as a list
        #can improve sampling strategy based on earlier papers
        samples=[str(row[0]) for row in df.select(col).dropna().distinct().limit(5).collect()]
        samples=",".join(samples)

        #profile would have ["Column:city|Type:str|Samples:Jersey City,Brooklyn"]
        profile.append(f"Column:`{col}`|Type:`{df.schema[col].dataType}`|Samples:[{samples}]")
        
    return "\n".join(profile)
@tool
@pandas_udf(ArrayType(FloatType()))
#use pandas udf for bulk processing, output is array - each element is array of floats
def generateEmbeddings(textInput:pd.series)->pd.series:
    '''Computes embeddings for text containing column name,data type,sample values'''
    model=SentenceTransform('sentence-transformers/all-MiniLM-L6-v2')
    embeddings=model.encode(textInput.tolist(),batch_size=32,show_progress_bar=False)
    return pd.series(embeddings.tolist())
    #returns [
    #           [for col 1: 0.1,0.2,0.4],  [for col 2: 0.1,0.2,0.4]
    #       ]


@tool
def cosineSimilarity(listA,listB)-> float:
    a,b=np.array(listA),np.array(listB)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm!=0 else 0.0


