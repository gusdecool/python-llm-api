API specs for endpoint /llm-job
Package it as in `app.route`

Check the task that is done
- [x] GET /llm-job return all. support query parameter:
    - limit = default 10, max 100
    - offset = 100
    - statuses = comma separated "queue", "processing", "done", "error" default "queue", "processing", "done" 
    - orderBy = e.g: created_at,desc
- [x] POST /llm-job create new llm job
- [x] PATCH /llm-job/:id update llm job, only allow update status & response
- [x] generate test for these endpoints