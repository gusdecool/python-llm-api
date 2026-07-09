API specs for endpoint /llm-job
Package it as in `app.route`

Check the task that is done
- [ ] GET /llm-job return all. support query parameter:
    - limit = default 10, max 100
    - offset = 100
    - statuses = comma separated "queue", "processing", "done", "error" default "queue", "processing", "done" 
    - orderBy = e.g: created_at,desc
- [ ] POST /llm-job create new llm job
- [ ] PATCH /llm-job/:id update llm job, only allow update status & response
- [ ] generate test for these endpoints