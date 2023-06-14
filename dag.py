from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
import sys, os

sys.path.append(os.getcwd())

from ocr.main import *

main = OCRMain()

default_args = {
    'owner': 'owner-name',
    'depends_on_past': False,
    'email': ['ourvideogroup@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

dag_args = dict(
    dag_id="tutorial-ocr-op",
    default_args=default_args,
    description='tutorial DAG ml',
    schedule_interval=timedelta(minutes=50),
    start_date=datetime(2023-06-14),
    tags=['example'],
)


with DAG( **dag_args ) as dag:
    start = BashOperator(
        task_id='start',
        bash_command='echo "start!"',
    )

    ready_task = PythonOperator(
        task_id='data_load',
        python_callable=main.ready_data
    )
    
    modeling_task = PythonOperator(
        task_id='train',
        python_callable=main.train_eval_data
    )


    complete = BashOperator(
        task_id='complete_bash',
        bash_command='echo "complete~!"',
    )

    start >> ready_task >> modeling_task >>  complete