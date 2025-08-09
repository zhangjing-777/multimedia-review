import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

dsn = os.getenv("DATABASE_URL") 

with open("init_sql.sql", "r", encoding="utf-8") as f:
    sql = f.read()

conn = None
try:
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("✅ 数据库初始化完成")
    
except Exception as e:
    if conn:
        conn.rollback()  # 回滚事务
    print(f"❌ 数据库初始化失败: {e}")
    
finally:
    if conn:
        conn.close()