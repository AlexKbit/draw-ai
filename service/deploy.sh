#Local run to check
#docker run -p8080:5000 -e PORT=5000 alexkbit/draw-ai:latest
heroku container:login
heroku container:push -a draw-ai-alexkbit web
heroku container:release web -a draw-ai-alexkbit