from mmrag.embed_shard import load
import torch

def embed_text(query):
    model, processor, device = load()

    with torch.no_grad():
        text_input = processor(text=query, return_tensors="pt", padding=True).to(device)
        embedded_text = model.get_text_features(**text_input)

        # if not tensor, retrieve tensor
        if not isinstance(embedded_text, torch.Tensor):   
            embedded_text = embedded_text.pooler_output

        # L2 normalization
        embedded_text = embedded_text / embedded_text.norm(dim=-1, keepdim=True)

    # move off gpu (in order to save) and convert to numpy array
    return embedded_text.cpu().numpy()

# retrieve closest num_hits hits from the table using the given query
def retrieve(table, query, num_retrievals=20):
    query_vector = embed_text(query)
    results = table.search(query_vector).limit(num_retrievals).to_list()

    return results
