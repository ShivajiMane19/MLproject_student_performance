# Machine Learning Project
## Student Performance 
1. Project Structure
2. Data Ingestion
3. EDA and Feature Engineering
4. Model Training
5. Model Deployment
6. Model Evaluation

# mlflow tracking
import dagshub
dagshub.init(repo_owner='maneshiva92', repo_name='MLproject_student_performance', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
