# AnimeWreck
An Anime Recommendation Engine, given the past watch history of the user with the ratings. Predicts the anime which the user can choose to watch.

Built by [Tofigh](https://github.com/tofigh-) & [Kshitij Karthick](https://github.com/KshitijKarthick)


## Features
* User can choose to be very specific to the watch history and expect only similar anime's or explore to find more popular ones which might not fall under the same theme or genre too.
* User can explore the anime embedding space to find other similar anime in a 3d Graph.
* User can choose to filter initial prospects which is passed to the model to contain genre embeddings or not (only similar anime)

## Deployment

```
docker-compose up
```

## [ Dev ] Serverside- FastAPI & Uvicorn

#### Project Setup
```
pip install -r requirements.txt

uvicorn server.main:app --reload --port 9000
```

## [ Dev ] Clientside - VueJs & Veutify

#### Project setup
```
npm install
```

#### Compiles and hot-reloads for development
```
npm run serve
```

#### Compiles and minifies for production
```
npm run build
```

#### Run your tests
```
npm run test
```

#### Lints and fixes files
```
npm run lint
```

#### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).

## Credits
* [Dataset](https://www.kaggle.com/azathoth42/myanimelist)
* [FastAi DL Library](https://docs.fast.ai/)
* [Pytorch](https://pytorch.org/)
* [Material Design Component Framework](https://vuetifyjs.com/en/)
* [Js Framework](https://vuejs.org/)
* [3d Force Graph](https://github.com/vasturiano/3d-force-graph)
* [Pandas](https://pandas.pydata.org/)

## Todo
* Need to add / convert preprocessing notebooks to Python files
* Model Resources / Models ?
