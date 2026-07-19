Goals:
Have an RAG agent that can add knowledge based on URL given and then can be queried by prompt.

For example, prompt: "Add https://www.kayak.co.id/ to rag knowledge base"

This will make agent_chooser to call the RAG agent to add the URL to the knowledge base. 
The RAG agent will then scrape the website and add the content to the knowledge base.
The information is saved in the sqlite database and can be queried by the RAG agent.

Then you can query the knowledge base by prompt: 
"What is kayak.co.id?" and the RAG agent will return the answer based on the knowledge it has added from the URL.

Another prompt:
"How kayak find such a low car rental price?" and the RAG agent will return the answer based on the knowledge it has added from the URL.