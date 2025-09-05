FROM nginx

WORKDIR /usr/share/react

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - 
RUN apt-get install -y nodejs

COPY RAG-frontend/package*.json ./

RUN npm install

COPY RAG-frontend/ .

RUN npm run build

RUN rm -r /usr/share/nginx/html/*

RUN cp -a dist/. /usr/share/nginx/html

EXPOSE 5173

COPY entrypoint.sh /usr/local/bin
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]