import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

@DBAcademyHelper.add_init
# Create the features table
def create_features_table(self):

    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.sql.functions import col
    from databricks.feature_engineering import FeatureEngineeringClient

    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS ca_housing")

    shared_volume_name = 'ca-housing'
    csv_name = 'ca-housing'
    
    # # Load in the dataset we wish to work on
    dataset_path = f"/Volumes/dbacademy_california_housing/v02/ca-housing/ca-housing.csv"

    feature_data_pd = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    
    # Add unique_id column
    feature_data_pd = feature_data_pd.withColumn("unique_id", monotonically_increasing_id())
    
    
    # Create the feature table using the PySpark DataFrame
    fe = FeatureEngineeringClient()
    table_name = f"{DA.catalog_name}.{DA.schema_name}.ca_housing"
    fe.create_table(
        name=table_name,
        primary_keys=["unique_id"],
        df=feature_data_pd,
        description="California Housing Feature Table",
        tags={"source": "bronze", "format": "delta"}
    )

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()
