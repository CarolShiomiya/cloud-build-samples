from google.cloud import storage as gcs
from google.oauth2 import service_account
import os
import datetime
import pytz

KEY_PATH = 'clove-v2-prd-263a649e0083.json'
PROJECT_ID = "clove-v2-prd"
BUCKET_NAME = "cardscan_model"


def connect_bucket():
    # モデルを置いているbucketに接続    
    credential = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = gcs.Client(PROJECT_ID, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    return(bucket)

def model_update():
    # ローカルにmodels/ディレクトリを作成
    os.makedirs("models", exist_ok=True)

    # モデルを置いているbucketに接続    
    bucket = connect_bucket()
    
    blob = bucket.blob("onepiece_aemodel/latest_ae_model.pth")
    blob.reload() 
    
    # クラウドのモデルの更新日時取得
    model_updated_dt = blob.updated
    japan_tz = pytz.timezone('Asia/Tokyo')
    model_updated_dt = model_updated_dt.astimezone(japan_tz).replace(tzinfo=None)
    print("クラウドの最新モデルは", model_updated_dt, "のものです。")

    local_model_path = "./models/latest_ae_model.pth"
    if os.path.exists(local_model_path):
        # ローカルのモデルの更新日時取得
        stat = os.stat(local_model_path)
        local_model_created_dt = datetime.datetime.fromtimestamp(stat.st_ctime)
        print("ローカルの最新モデルは", local_model_created_dt, "のものです。")
    
        if local_model_created_dt < model_updated_dt:
            print("モデルが最新ではありません。ダウンロードを開始します。")
            b = blob.download_to_filename("./models/latest_ae_model.pth")

        else:
            print("現在のモデルは最新です。")
            
    else:
        print("モデルがありません。ダウンロードを開始します。")
        b = blob.download_to_filename("./models/latest_ae_model.pth")



def dict_update():
    # ローカルにdicts/ディレクトリを作成
    os.makedirs("dicts", exist_ok=True)

    # モデルを置いているbucketに接続    
    bucket = connect_bucket()
    
    blob = bucket.blob("onepiece_dictionary/latest_dict.json")
    blob.reload() 
    
    # クラウドのモデルの更新日時取得
    dict_updated_dt = blob.updated
    japan_tz = pytz.timezone('Asia/Tokyo')
    dict_updated_dt = dict_updated_dt.astimezone(japan_tz).replace(tzinfo=None)
    print("クラウドの最新辞書は", dict_updated_dt, "のものです。")

    local_dict_path = "./dicts/latest_dict.json"
    if os.path.exists(local_dict_path):
        # ローカルのモデルの更新日時取得
        stat = os.stat(local_dict_path)
        local_dict_created_dt = datetime.datetime.fromtimestamp(stat.st_ctime)
        print("ローカルの最新辞書は", local_dict_created_dt, "のものです。")
    
        if local_dict_created_dt < dict_updated_dt:
            print("辞書が最新ではありません。ダウンロードを開始します。")
            b = blob.download_to_filename("./dicts/latest_dict.json")

        else:
            print("現在の辞書は最新です。")
            
    else:
        print("辞書がありません。ダウンロードを開始します。")
        b = blob.download_to_filename("./dicts/latest_dict.json")


def app_update(prefix="cardscanner_app/", local_path=""):
    # 指定されたバケットを取得
    bucket = connect_bucket()

    # バケット内のファイル一覧を取得
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        # ダウンロード先のローカルファイルパスを構築
        local_file_path = os.path.join(local_path, blob.name)
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # クラウドのファイルの更新時刻
        blob.reload()
        file_updated_dt = blob.updated
        japan_tz = pytz.timezone('Asia/Tokyo')
        file_updated_dt = file_updated_dt.astimezone(japan_tz).replace(tzinfo=None)

        # ファイルをダウンロード（ディレクトリではないことを確認）
        if not blob.name.endswith('/'):
            # すでにローカルにあるかどうか判定
            if os.path.exists(local_file_path):            
                # ローカルファイルの更新日時取得
                stat = os.stat(local_file_path)
                local_file_created_dt = datetime.datetime.fromtimestamp(stat.st_ctime)
                if local_file_created_dt < file_updated_dt:
                    blob.download_to_filename(local_file_path)
                    print(f"Update {local_file_path}")

            else:
                blob.download_to_filename(local_file_path)
                print(f"Downloaded {blob.name} to {local_file_path}")
