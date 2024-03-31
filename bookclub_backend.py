import streamlit as st
import sqlite3
import hashlib

#one database driver within app
def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance

class DatabaseDriver(object):
    """
    Databse driver for the bookclub app.
    Handles with reading and writing data
    within the database
    """
    def __init__(self):
        """
        Secure a connction w/ the database
        and store it into the instance 'c'
        """
        # Connect to the SQLite database
        self.c = sqlite3.connect('bookclub.db', check_same_thread = False)

        #Debugging purposes I reset the tables 
        self.c.execute("DROP TABLE IF EXISTS Users;")
        self.c.execute("DROP TABLE IF EXISTS Books;")
        self.c.execute("DROP TABLE IF EXISTS BookReadingHistories;")
        self.c.execute("DROP TABLE IF EXISTS ChatHistories;")
        self.c.execute("DROP TABLE IF EXISTS Chapters")
        self.c.execute("DROP TABLE IF EXISTS BooksHistoryAssociation;")

        self.create_user_table()
        self.create_book_library_table()
        self.create_book_history_table()
        self.create_chat_history_table()
        self.create_library_history_assoctable()

        #Testing purposes


    def create_user_table(self):
        """
        Create a database of users
        no explicit relationship
        """
        try:
            self.c.execute('''
            CREATE TABLE IF NOT EXISTS Users(
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            );
            ''')
        except Exception as e:
            print("Error creating table: ", e)
            raise e

    def create_chapter_table(self):
        """
        Create a table of Chapters
        explicit: Book (one to many) Chapters
        """
        try:
            self.c.execute('''
            CREATE TABLE IF NOT EXISTS Chapters(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_number INTEGER NOT NULL,
                chapter_title TEXT NOT NULL,
                chapter_text TEXT NOT NULL,
                book_id INTEGER NOT NULL,
                FOREIGN KEY(book_id) REFERENCES Books(id)
            );
            ''')
        except Exception as e:
            print("Error creating Chapters table: ", e)
            raise e

    def create_book_library_table(self):
        """
        Create the books table if it doesn't exist
        Many to many relationship with Book reading histories
        """
        try:
            self.c.execute(
            '''CREATE TABLE IF NOT EXISTS Books(
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                author TEXT NOT NULL,
                title TEXT NOT NULL,
                pages INTEGER NOT NULL
                );
            ''')
        except Exception as e:
            print("Error creating Books table: ", e)
            raise e
        
    def create_chat_history_table(self):
        """
        Create a chat history table
        User(one to many) Chat Histories
        BookReadingHistories(one to one)Chat Histories
        """
        # Chat History table
        try:
            self.c.execute('''
                CREATE TABLE IF NOT EXISTS ChatHistories(
                    id INTEGER PRIMARY KEY,
                    time DATETIME NOT NULL,
                    reading_id INTEGER NOT NULL UNIQUE,
                    user_id, TEXT NOT NULL,
                    FOREIGN KEY (reading_id) REFERENCES BookReadingHistories(id),
                    FOREIGN KEY (user_id) REFERENCES Users(id)
                );
            ''')
        except Exception as e:
            print("Error creating chat histories table: ", e)
            raise e


    def create_book_history_table(self):
        """
        Book Reading History table
        Many to Many relationship with books
        """
        
        try:
            self.c.execute(
            '''CREATE TABLE IF NOT EXISTS BookReadingHistories(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_title TEXT NOT NULL,
                current_chapter INTEGER NOT NULL,
                current_page INTEGER NOT NULL,
                time DATETIME NOT NULL
            );
            ''')
        except Exception as e:
            print("Error creating Book Histories table: ", e)
            raise e

    def create_library_history_assoctable(self):
        """
        Association table between book library and
        books 
        """
        try:
            self.c.execute('''
                CREATE TABLE IF NOT EXISTS BooksHistoryAssociation(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL,
                    reading_id INTEGER NOT NULL,
                    FOREIGN KEY (book_id) REFERENCES Books(id),
                    FOREIGN KEY (reading_id) REFERENCES BookReadingHistories(id)
                );'''
            )
        except Exception as e:
            print("Error creating Association table: ", e)
            raise e
    
    def get_all_books(self):
        """
        Get all books in the database
        """
        cursor = self.c.execute("SELECT * FROM Books;")
        books = []
        #Retrieve data from the database
        for row in cursor:
            books.append({
                "Avatar Name": row[1],
                "Book Contents": row[4],
                "Voice Speed": row[5],
                "Reference Code": row[6]
            })
        return books

    #Getting Most Recent Book History 
    # user id --> most recent chat history --> return most recent book history ?
    
    def get_user_by_id(self, user_id):
        """
        Return a user based on the session id given
        """
        cursor = self.c.execute("SELECT * FROM Users WHERE id = ?", (user_id,))
        for row in cursor:
            return ({"id": row[0], "name": row[1]})
        return None 
    
    def get_chat_histories(self, user_id = None, chat_id = None):
        """
        Get all chat histories based on the user id or chat_id given
        """
        cursor = None
        if(user_id != None):
            cursor = self.c.execute("SELECT * FROM ChatHistories WHERE user_id = ?", (user_id))
        elif(chat_id != None):
            cursor = self.c.execute("SELECT * FROM ChatHistories WHERE id = ?", (chat_id))
        
        if cursor == None:
            return None
           
        chs = []
        for row in cursor:
            chs.append({"id": row[0], "timestamp": row[1], "reading id": row[2]})
        return chs

    def get_book_history(self, book_id = None, bh_id = None ):
        """
        Return a book history based on a given book id or book history id
        """
        cursor = None
        if bh_id != None:
            cursor = self.c.execute("SELECT * FROM BookHistories WHERE id = ?", (bh_id))
        if book_id != None:
            cursor = self.c.execute("SELECT * FROM BookHistories WHERE book_id = ?", (book_id))
        if cursor == None:
            return "Error", None
        bhs = []
        for row in cursor:
            bhs.append({"id": row[0], "book title": row[1], "current chapter": row[2], "current page": row[3], "timestamp": row[4]})
        return bhs
    

    def create_user(self, session_id, username, password):
        """
        Create a user
        Requires an username and password
        """
        cursor = self.c.cursor()

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        cursor.execute('INSERT INTO Users (id, username, password) VALUES (?, ?, ?)', (session_id, username, hashed_password))

        self.c.commit()
        
        return session_id

    def login_user(self, username, password):
        """
        Login an user
        Requires username and password
        """
        cursor = self.c.cursor()

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_password))
        user = cursor.fetchone()

        if user:
            return True, user
        else:
            return False, None 
        
    #TODO: add more create stuff 
    

DatabaseDriver = singleton(DatabaseDriver)
