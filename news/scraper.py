import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_news_html(query: str, date: str) -> str:
    """
    Realiza la solicitud a Google News y retorna el contenido HTML.
    """
    url = f'https://news.google.com/search?q={query}%20{date}&hl=en-US&gl=US&ceid=US:en'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error al obtener noticias del {date}: {e}")
        return ""

def parse_news_and_dates(html_content: str):
    """
    Parsea el HTML y retorna listas de textos y fechas.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = soup.find_all(class_='JtKRv')     # Clase para títulos
    dates = soup.find_all(class_='hvbAAd')     # Clase para fechas

    texts = [title.text for title in titles]
    datetimes = [dt.get('datetime') for dt in dates]

    return texts, datetimes

def get_news_by_date_range(query: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Recolecta noticias por rango de fechas y retorna un DataFrame con columnas 'news' y 'date'.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    all_news = []
    all_dates = []

    for day in tqdm(range((end - start).days + 1), desc="Scraping Google News"):
        date_str = current.strftime("%Y-%m-%d")
        html = get_news_html(query, date_str)

        if html:
            news, dates = parse_news_and_dates(html)
            all_news.extend(news)
            all_dates.extend(dates)

        time.sleep(1)  # Evitar bloqueos de Google
        current += timedelta(days=1)

    df = pd.DataFrame({'news': all_news, 'date': all_dates})
    return df.dropna()

def save_news_to_parquet(df: pd.DataFrame, filename: str):
    """
    Guarda el DataFrame en formato Parquet.
    """
    df.to_parquet(filename, index=False)
    logging.info(f"Se guardaron {len(df)} noticias en {filename}")



if __name__ == "__main__":
    df_news = get_news_by_date_range("NVIDIA", "2022-01-01", "2024-12-31")
    save_news_to_parquet(df_news, "nvidia_news.parquet")
