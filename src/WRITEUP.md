# Approach
## The 4 main components
* The client.
* The powerfull, expensive LLM. For testing I used the remote gemini-2.5-flash, however any equivalently powerfull model will do.
* The weak, fast, local LLM used for embeddings. I used all-MiniLM-L6-v2, for a good balance of speed and accuracy.
* The database containing the values of the embedings of the companies. LanceDB was chosen for this task because it is optimized for quick retreivals based on vector data. Supports clustering based on cosine simmilarity.
* The backend linking everything together.

## Componnt interaction
* Every company description provided is embedded in the space of the weak LLM and saved in the database in a separate script (database_setup.py), which only runs if the database needs an update.
* The client inputs the query.
* The query is sent to the powerfull LLM which generates a hypothetical description of a company matching the clients request. The description is provided as a JSON simmilar in format to the ones provided.
* The description provded by the LLM is compared against other descriptions (the full JSON) in the database, by cosine simmilarity.
* The best 50 matches are returned sorted to the client.

## Design motivation
### Efficiency & Performance
* Each query only uses one remote LLM request, for generating the description, so just one large request per query.
* The local model is small enough for the computations to be fast when a relatively small number of compaies must be returned (< 1000), thus it can run locally on any modern general use device without issues.
* LanceDB provides clustering, so if the company database becomes very large, only a small fraction of the companies are actually checked, allowing the system to scale effectively. It will however take a long time to calculate the initial embeddings.

### Accuracy
* The limitations of cosine search are mitigated by the request to the powerfull LLM which can use its strong semantic understanding to provide an accurate description. This covers the case mentioned in the task description, wherea query like "Packaging forcosmetic items" would return cosmetics companies.

# Tradeoffs
The system was optimized for simplicity, in the detriment of robustness. The modular architecture makes it easy to change the LLM providers. The high reliance on the powerfull LLM to provide an accurate description is a potential point of failure, however considering the scope of the system it is an acceptable compromise.
Other elements like speed, cost and accuracy were balanced among each other, however these can be easily modified by changing the 2 LLMs.

# What works well
The system handles specific requests very well. This is thanks to it comparing the entire JSON, so no details are ommited. At first it only used the Description field in the json, however upon further testing, the full JSON approach produced better results. For example

>B2B SaaS companies providing HR solutions in Europe

returns very accurate results.

# Error Analysis
The system tends to struggle with undetailed and ambiguous queries. This happens due to the limitations of the powerfull LLM, which makes assumptions based on what the client requested, some of which beeing irrelevant. A concrete example is the query:

> Companies helping people find homes in europe

Instead of returning real estate companies it favours software developement companies, particularly the ones providing HR services. This happens because the powerfull LLM produces a description like 

> EuroHomeSeek Solutions is a digital platform and consultancy service dedicated to simplifying the home-finding process for individuals, families, and professionals relocating within or to Europe. We leverage technology and local expertise to connect users with suitable properties, manage logistics, and provide comprehensive relocation support, ensuring a smooth transition to their new European home.

Which emphasizes the digital and consulting aspects, rather than the real estate one.

# Technical problems
In this exact form the system is limited by the availability to the gemini-2.5-flash model offered by the free tier of Google AI Studio. If the model is under high demand, 503 UNAVAILABLE HTTP errors are very common. It also has a limited number of tokens per day.

This can easily be fixed however with premium plans, or by runing the powerfull model on company servers.

# Scaling
Scaling is handled by LanceDB. The key line is: ``` table.create_index(metric="cosine") ```

This automatically clusters simmilar entries so each query will only search the closest clusters, containig the simmilar entries, thus avoiding a lot of unnecessarry computations. This approach does remove the guarantee that the closest entries in terms of cosine distance are returned, however the accuracy loss is very low, compared to the huge performance boost.

# Failure modes
The largest failure point is the powerfull LLM. A mistakenly generated description is enough to ruin the query entirely, and the chance of this happening is larger the more ambiguous the clients request is.

The incorrect results appear to be strongly correlated to low scores (large cosine distances for even the best matches), so the best approach would be to monitor these scenarios in production and work out a solution based on the data.

# Other notes
* Currently all the embeddings are recalculated if even one changes. This should be changed for effective scaling.
* This is currently a console application. For real clients a UI should be built, so the users can easily interact with the app, however this was not implemented because I considered it to be beyond the scope of this project.