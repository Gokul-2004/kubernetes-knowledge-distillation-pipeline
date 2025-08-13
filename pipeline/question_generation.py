from openai import OpenAI
import os
from loguru import logger
from typing import List, Dict, Any

def get_selected_topics() -> str:
    """Get the selected topics from environment variable or use default"""
    selected_topics = os.environ.get('SELECTED_TOPICS', 'Kubernetes Architecture, Pods and Containers, Services and Networking')
    return selected_topics

def generate_questions(knowledge_base: str, num_questions: int = 1) -> List[str]:
    """
    Generate questions based on the knowledge base and selected topics
    
    Args:
        knowledge_base: The knowledge base text
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    try:
        # Get selected topics
        selected_topics = get_selected_topics()
        
        # Create the prompt with selected topics
        prompt = f"""Based on the following Kubernetes v1.28 knowledge base and the selected topics: {selected_topics}

Knowledge Base:
{knowledge_base}

Generate {num_questions} diverse and challenging questions that:
1. Are specifically about Kubernetes v1.28 concepts and features
2. Cover the selected topics: {selected_topics}
3. Can be answered using the knowledge base information
4. Cover different aspects and difficulty levels (basic to advanced)
5. Test understanding of practical Kubernetes concepts
6. Are clear and well-formulated

Generate exactly {num_questions} questions, one per line, without numbering or bullet points."""

        # Make API call to OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert Kubernetes question generator. Generate clear, relevant questions based on the provided Kubernetes v1.28 knowledge base and topics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract questions from response
        questions_text = response.choices[0].message.content.strip()
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        # Ensure we have the right number of questions
        if len(questions) > num_questions:
            questions = questions[:num_questions]
        elif len(questions) < num_questions:
            # Generate additional questions if needed
            additional_needed = num_questions - len(questions)
            additional_prompt = f"""Generate {additional_needed} more questions based on the topics: {selected_topics}

Make sure the questions are different from these existing questions:
{chr(10).join(questions)}

Generate exactly {additional_needed} additional questions, one per line."""
            
            additional_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Kubernetes question generator. Generate clear, relevant questions based on the provided Kubernetes topics."},
                    {"role": "user", "content": additional_prompt}
                ],
                max_tokens=300,
                temperature=0.8
            )
            
            additional_questions = [q.strip() for q in additional_response.choices[0].message.content.strip().split('\n') if q.strip()]
            questions.extend(additional_questions[:additional_needed])
        
        logger.info(f"Generated {len(questions)} questions.")
        return questions
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        # Fallback questions based on selected topics
        selected_topics = get_selected_topics()
        fallback_questions = [
            f"What are the key principles of {selected_topics.split(',')[0].strip()} in Kubernetes v1.28?",
            f"How does {selected_topics.split(',')[1].strip() if len(selected_topics.split(',')) > 1 else selected_topics.split(',')[0].strip()} work in Kubernetes?",
            f"What are the main applications of {selected_topics.split(',')[2].strip() if len(selected_topics.split(',')) > 2 else selected_topics.split(',')[0].strip()} in Kubernetes v1.28?"
        ]
        return fallback_questions[:num_questions]

def generate_reference_answers(questions: List[str], knowledge_base: str) -> List[str]:
    """
    Generate reference answers using GPT-4o-mini
    
    Args:
        questions: List of questions to answer
        knowledge_base: The knowledge base text
        
    Returns:
        List of reference answers
    """
    try:
        reference_answers = []
        
        for question in questions:
            prompt = f"""Based on the following Kubernetes v1.28 knowledge base, provide a comprehensive and accurate answer to this question:

Knowledge Base:
{knowledge_base}

Question: {question}

Provide a detailed, well-structured answer that directly addresses the question using information from the Kubernetes v1.28 knowledge base. Focus on practical Kubernetes concepts and features."""

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Kubernetes specialist answering questions based on provided Kubernetes v1.28 knowledge. Provide comprehensive, accurate answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            reference_answers.append(answer)
        
        logger.info(f"Generated {len(reference_answers)} reference answers.")
        return reference_answers
        
    except Exception as e:
        logger.error(f"Error generating reference answers: {str(e)}")
        # Fallback answers
        fallback_answers = [
            "This is a reference answer based on the Kubernetes v1.28 knowledge base. The specific answer would depend on the question and available Kubernetes information.",
            "Based on the Kubernetes knowledge base, this question can be answered using the provided information and relevant Kubernetes concepts.",
            "The Kubernetes v1.28 knowledge base contains information that can be used to formulate a comprehensive answer to this question."
        ]
        return fallback_answers[:len(questions)] 