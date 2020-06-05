#Local run to check
heroku container:login
heroku container:push -a draw-ai-alexkbit web
heroku container:release web -a draw-ai-alexkbit