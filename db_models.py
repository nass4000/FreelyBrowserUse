from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    plans = relationship("Plan", back_populates="session", cascade="all, delete-orphan")
    search_results = relationship("SearchResult", back_populates="session", cascade="all, delete-orphan")
    scraped_pages = relationship("ScrapedPage", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("Session", back_populates="messages")

class Plan(Base):
    __tablename__ = 'plans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    queries = Column(JSON)
    steps = Column(JSON)
    rationale = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("Session", back_populates="plans")

class SearchResult(Base):
    __tablename__ = 'search_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    query = Column(String)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("Session", back_populates="search_results")

class ScrapedPage(Base):
    __tablename__ = 'scraped_pages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('sessions.id'))
    url = Column(String)
    title = Column(String)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("Session", back_populates="scraped_pages")

class HtmlCache(Base):
    __tablename__ = 'html_cache'
    id = Column(String, primary_key=True)
    html = Column(Text)
    url = Column(String)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///freely_browser.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()