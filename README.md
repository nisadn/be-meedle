This is a Django project.

## About Meedle

This is a college assignment for Information Retrieval course. Meedle is a medical search engine that uses collections from Medline and BM25 as the scoring method. The project consisted of two repositories. This is my backend repo and the other one is for [frontend](https://github.com/nisadn/meedle).

## List of Endpoints

### Query retrieval

`POST /search_query`

Request body

```
{
    "query": "alkylated with radioactive iodoacetate",
    "k": 10
}
```

### Get docs

`POST /get_docs`

Request body

```
{
    "docs_id": [
        "6\\507.txt",
        "6\\554.txt",
        "11\\1003.txt"
    ],
    "truncate": true
}
```

## Run Locally

Install the dependencies once with `python -m pip install -r requirements.txt` 

Run the development server with `python manage.py runserver`

Open [http://localhost:8000](http://localhost:8000) with your browser to see the result.

## Deployed on Vercel

Check out Meedle Backend [here](https://be-meedle.vercel.app/).
