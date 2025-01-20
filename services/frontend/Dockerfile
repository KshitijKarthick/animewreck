FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y curl build-essential nginx \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs

WORKDIR /website

COPY package.json babel.config.js package-lock.json postcss.config.js vue.config.js .eslintrc.js /website/

RUN npm config set strict-ssl false \
    && npm install

COPY src /website/src

RUN npm run build

# # COPY src/robots.txt src/sitemap.xml /website/dist/

EXPOSE 80

STOPSIGNAL SIGTERM

RUN rm -r /usr/share/nginx/html \
    && ln -s /website/dist /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]
