heroku container:login
heroku container:push alexkbit/draw-ai --app draw-ai-alexkbit
heroku container:release web --app draw-ai-alexkbit