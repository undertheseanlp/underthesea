# Setup server

## Geting Started

Install dependencies

```
pip install -r requirements.txt
```

Initialize database

```
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser # Username: admin, password: 123456
```

## Run Tests

```
cd backend
python manage.py test languages/tests
```

## Run server

Run server 

```
python manage.py runserver
```

Use Application

* Go to http://localhost:8000/admin
* Login with username and password
* Go to http://localhost:8000/api
