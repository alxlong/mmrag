from mmrag.generate import to_b64
import anthropic, os

def score_retrieval(retrieved_keys, relevant_keys, k=5):
    num_accurate_hits = len(set(retrieved_keys) & set(relevant_keys))

    precision_at_k = num_accurate_hits / k
    recall = num_accurate_hits / len(relevant_keys) if relevant_keys else 0
    rr = next((1/(i+1) for i, key in enumerate(retrieved_keys) if key in set(relevant_keys)), 0)

    return {'hits': num_accurate_hits, 'precision_at_k': precision_at_k, 'recall': recall, 'rr': rr}

def judge_answer(question, answer, images):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    content = [
                {'type': 'image',
                'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': to_b64(img)}}
                for img in images
              ]

    content.append({
                    'type': 'text', 
                    'text': f'''You are evaluating whether an answer is supported by the images provided as evidence. 

                            QUESTION: {question} 

                            ANSWER: {answer} 

                            The attached images are the only evidence available. 

                            Decompose the answer into individual claims. For each, decide whether it 
                            is verifiable from at least one attached image. 

                            Rules: 
                            - Contextual claims count — time of day, weather, season, location require visual evidence. 
                            - A claim is supported if any single image supports it. 
                            - When images conflict, an answer reporting one value without acknowledging variation is partially grounded. 
                            - Inferred specifics (model year, brand, breed, names) are unsupported unless legible or unambiguous. 
                            - Claims true in general but not visible are unsupported. 
                            - Borderline claims are unsupported. 

                            Return the same struncture as a JSON but do NOT format it as JSON:
                            {{"claims": [{{"claim": "...", "supported": true, "reason": "..."}}], 
                              "verdict": "fully_grounded" | "partially_grounded" | "ungrounded" | "refusal"}}''' 
                    })

    message_list = [{'role': 'user', 'content': content}]
    
    response = client.messages.create(model='claude-sonnet-4-5', max_tokens=1024, messages=message_list)
    
    return response.content[0].text
