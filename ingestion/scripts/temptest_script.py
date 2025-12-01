import pinecone
import os
PINECONE_API_KEY="pcsk_C9Zww_QEXcH1khKnkDHkbJhjW9hBwsDWNVJu5hZL4ivNsZ6hDTwPtkiWBiFenM8WzebRy"
pc = pinecone.Pinecone(PINECONE_API_KEY)
index = pc.Index("company-docs")

# fetch one vector
res = index.fetch(ids=["503f8c92f5466a04204615463e60c85f"])
print(res)
print(len(res.vectors["503f8c92f5466a04204615463e60c85f"].values))
