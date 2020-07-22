input_df = spark.read.option("header", "true").csv(s3_input_data_path)
 
rearranged_col_names_df = input_df.select(*columns)
 
# drop null values
cleaned_df = rearranged_col_names_df.dropna()
print("Dropped null values")
 
# split dataframe into train and validation
splits = cleaned_df.randomSplit([0.7, 0.3], 0)
print("Split data into train and validation")
 
train_df = splits[0]
validation_df = splits[1]
 
train_data_output_path = f'{s3_processed_data_path}/train'
validation_data_output_path = f'{s3_processed_data_path}/validation'
 
print(f"Train data output path: {train_data_output_path}")
print(f"Validation data output path: {validation_data_output_path}")
 
# write data to S3
train_df.coalesce(1).write.csv(train_data_output_path, mode='overwrite', header=False)
validation_df.coalesce(1).write.csv(validation_data_output_path, mode='overwrite', header=False)