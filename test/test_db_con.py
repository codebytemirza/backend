import mysql.connector

conn = mysql.connector.connect(
    host="66.102.139.243",
    user="theneutralai_admin",
    password="yy+SYGTNpC9zJ5+k",
    database="theneutralai_login_system"
)

print("Connected!")
cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
for table in cursor:
    print(table)
conn.close()
