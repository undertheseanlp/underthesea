# README

Run flake8 before commit

```
black .
flake8 .  --max-complexity 10 --ignore E501,W504,W605
```