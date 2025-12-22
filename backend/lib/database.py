import asyncio
import os

import pandas as pd
from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
CSV_FILE = "train/data/full_markup.csv"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class DataSample(Base):
    __tablename__ = "geoakinator_data"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)


async def export_data(path="data/actual_ds.csv"):
    async with AsyncSessionLocal() as session:
        result = await session.execute(DataSample.__table__.select())
        rows = result.fetchall()

    df = pd.DataFrame(rows, columns=DataSample.__table__.columns.keys())
    df.to_csv(path, index=False)
    return path


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    df = pd.read_csv(CSV_FILE)
    df.dropna(subset=["description", "lat", "lng"], inplace=True)
    async with AsyncSessionLocal() as session:
        for _, row in df.iterrows():
            item = DataSample(text=row["description"], lat=row["lat"], lon=row["lng"])
            session.add(item)
        await session.commit()


if __name__ == "__main__":
    asyncio.run(init_db())
