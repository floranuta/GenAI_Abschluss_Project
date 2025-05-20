import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from tavily_agent import tavily_search_with_date

# 🔄 Lädt Umgebungsvariablen aus der .env-Datei
load_dotenv()
# Проверка (временно добавь это!)
#print("NEWS_API_KEY:", os.getenv("NEWS_API_KEY"))
#print("ALPHA_VANTAGE_API_KEY:", os.getenv("ALPHA_VANTAGE_API_KEY"))

COMPANIES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOG",
    "nvidia": "NVDA",
    "meta": "META"
}

TRUSTED_SOURCES = {
    "cnbc", "bloomberg", "reuters", "yahoo", "financial times", "marketwatch"
}

SCRAPING_SOURCES = {
    "CNBC": {
        "base_url": "https://www.cnbc.com/search/?query={query}",
        "item_selector": "a.Card-title",
        "title_selector": "a.Card-title",
        "link_prefix": ""
    },
    "Yahoo": {
        "base_url": "https://finance.yahoo.com/quote/{query}/news",
        "item_selector": "li.js-stream-content h3 a",
        "title_selector": "li.js-stream-content h3 a",
        "link_prefix": "https://finance.yahoo.com"
    },
    "MarketWatch": {
        "base_url": "https://www.marketwatch.com/search?q={query}",
        "item_selector": "h3.article__headline a",
        "title_selector": "h3.article__headline a",
        "link_prefix": ""
    }
}

def generic_scrape(source_name, config, query, max_articles=3):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = config["base_url"].format(query=query)
    print(f"Scraping URL: {url}")  # Debug-Ausgabe
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select(config["item_selector"])[:max_articles]
        results = []
        print(f"Anzahl gefundener Artikel: {len(items)}")  # Debug-Ausgabe
        for item in items:
            tag = item.select_one(config["title_selector"])
            if tag and tag.text.strip():
                title = tag.text.strip()
                print(f"Scraped title: {title}")  # Debug-Ausgabe
                link = tag.get("href", "#")
                full_link = link if link.startswith("http") else config["link_prefix"] + link
                results.append((None, f"[{source_name}] {title} ({full_link})"))

        return results if results else [(None, f"[{source_name}] Keine Ergebnisse.")]
    
    except Exception as e:
        return [(None, f"Fehler beim Scraping von {source_name}: {e}")]

def scrape_all_sources(query, max_articles=3):
    all_results = []
    for name, config in SCRAPING_SOURCES.items():
        all_results += generic_scrape(name, config, query, max_articles)
    return all_results

""" def is_news_related_to_company(article, company_name, ticker):
    title = article.get("title", "").lower()
    description = article.get("description", "").lower()
    content = title + " " + description
    source = article.get("source", {}).get("name", "").lower()

    # 🔍 Простое совпадение
    basic_match = company_name.lower() in content or ticker.lower() in content 

    # 🔎 Проверка доверенного источника
    trusted = any(src in source for src in TRUSTED_SOURCES)
    
    return basic_match  and trusted """

# 📰 Funktion: Abrufen aktueller Nachrichten über NewsAPI
def get_latest_news(query,ticker,max_articles=5):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return ["NEWS_API_KEY fehlt."]
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query, 
        "apiKey": api_key, 
        "language": "en",
        "sortBy": "publishedAt"
        }
    try:
        response = requests.get(url, params=params)
        articles = response.json().get("articles", [])
        results = []
        for a in articles[:max_articles]:
            title = a.get("title", "")
            description = a.get("description", "")
            source = a.get("source", {}).get("name", "")
            url = a.get("url", "")
            #if  is_news_related_to_company(a, query, ticker):
            results.append(f"- {title}\n {description}\n ({source})\n {url}\n")
        return results if results else ["Keine relevanten Artikel gefunden."]    
    except Exception as e:
        return [f"Fehler bei NewsAPI: {e}"]

# 💹 Funktion: Aktueller Aktienkurs über Yahoo Finance (yfinance)
def get_stock_price_yahoo(ticker):
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="5d")
        if hist.empty:
            return f"Keine Daten für {ticker}."
        last = hist.iloc[-1]
        return f"Schlusskurs am {last.name.date()}: {round(last['Close'], 2)} USD"
    except Exception as e:
        return f"Yahoo Finance Fehler: {e}"

# 📊 Funktion: Abrufen von Zeitreihendaten über Alpha Vantage API    
def get_alpha_vantage_data(symbol):
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "ALPHA_VANTAGE_API_KEY fehlt."
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": api_key}
    try:
        response = requests.get(url, params=params)
        data = response.json().get("Time Series (Daily)", {})
        if not data:
            return "Keine Daten von Alpha Vantage."
        latest_date = sorted(data.keys())[-1]
        close_price = data[latest_date]["4. close"]
        return f"{symbol} (Alpha Vantage) Schlusskurs am {latest_date}: {close_price} USD"
    except Exception as e:
        return f"Alpha Vantage Fehler: {e}"




# 🧠 Hauptfunktion des Web-Agenten
def handle_company(name):
    ticker = COMPANIES.get(name.lower())
    if ticker is None:
        ticker = COMPANIES.get(name.capitalize())
    if not ticker:
        print("❌ Unbekannte Firma. Verfügbare Firmen:")
        for cname in COMPANIES.keys():
            print(f"- {cname.capitalize()}")
        return

    print(f"\n📊 Daten für {name.capitalize()} ({ticker})")

    print("\n🔎 Aktuelle Nachrichten via News_API:")
    for news in get_latest_news(name,ticker):
        print(news)

    """ print("\n🌐 Web Scraping:")
    for news in scrape_all_sources(name, max_articles=3):
        print(news) """

    print("\n💰 Yahoo Finance:")
    print(get_stock_price_yahoo(ticker))

    print("\n📈 Alpha Vantage:")
    print(get_alpha_vantage_data(ticker))

    #new_query = input("Frag etwas zum Markt (z.B. 'Was gibt es Neues bei NVIDIA?'): ")
    print("\n🌍 Tavily (mit Datum):")
    for result in tavily_search_with_date(f"{name} stock news"):
        print(result)

def smart_handle(query):
    # Suche nach dem Firmennamen im Query
    for name, ticker in COMPANIES.items():
        
        if name.lower() in query.lower():
            print(f"📦 Anfrage erkannt für: {name.title()} ({ticker})")
            return handle_company(name)
    return "❌ Keine bekannte Firma im Text gefunden."

# Interaktive Eingabe
#if __name__ == "__main__":
#    company_input = input("Gib den Namen einer Firma ein (Apple, Microsoft, Google, Nvidia, Meta): ")
#    handle_company(company_input)