import cx_Oracle as cx

class DAO:
    def __init__(self, host, database, user, password, port):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        
    def __enter__(self):
        self.conn = cx.connect(host=self.host,
                               database=self.database,
                               user=self.user,
                               passwd=self.password,
                               port=self.port)
        return self.conn
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.conn.close()
        
    def execute(self, sql):
        with DAO(self.host, self.database, self.user, self.password, self.port) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            row = cursor.fetchall()
            return row