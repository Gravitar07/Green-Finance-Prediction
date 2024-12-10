from groq import Groq
from .exception import log_exception, APIError
from .config import Config
from .logger import get_logger

logger = get_logger(__name__)

class LLM:
    def __init__(self):
        try:
            # Configure and initialize the Generative AI model
            self.llm = Groq(api_key=Config.GROQ_API_KEY)
            logger.info("LLM model initialized successfully.")
        except Exception as e:
            log_exception(e, "Failed to initialize the LLM model in LLM class.")
            raise APIError("Error initializing LLM model. Check API key and model name.") from e

    def prompt_template(self, result):
        # Generates a structured prompt for the LLM based on the provided data
        prompt = f"""You are a Green Finance Advisor specializing in evaluating companies based on sustainability impact areas, 
    green finance readiness, and risk probabilities predicted by an ML model. Your role is to analyze the provided 
    data, including ML output, to generate a comprehensive, actionable report highlighting the company's strengths, 
    areas for improvement, risk levels, and certification recommendations.

    **Instructions:**
    1. Analyze the provided data, including the ML Model's Risk Probability, to create a structured Green Finance Diagnostic Report.
    2. Integrate the ML output into your assessment, particularly in identifying risks and areas for improvement.
    3. Ensure the report is professional, clear, and actionable for decision-makers.
    4. Avoid technical jargon and maintain accessibility while ensuring accuracy.

    **Input Data**:
    {result}

    **Output Format**:

    1. **Company Overview**:
        - Provide a brief summary of the company, including its name, location, industry, and product/service offerings.
        Example:
        "The company GreenTech Solutions operates in the Renewable Energy sector in Germany, providing solar and wind power solutions to industrial clients."

    2. **ML Risk Probability**:
        Highlight the ML-predicted risk probability and interpret its significance.
        Example:
        - Risk Probability: 0.72 (High Risk)
        "The ML model predicts a 72% probability of sustainability-related risks, indicating a need for focused interventions in governance and environmental impact areas."

    3. **Sustainability Impact Areas**:
        Assess the company's performance in the following areas:
        - **Community**: Highlight initiatives benefiting local communities, such as employment generation or educational programs.
        - **Environment**: Discuss environmental practices, like emissions reductions or renewable energy use.
        - **Customers**: Evaluate customer-focused sustainability measures, such as product lifecycle assessments or ethical sourcing.
        - **Governance**: Examine the company's governance practices, including transparency and ethical policies.

    4. **Key Strengths**:
        Identify and summarize the company's strongest green finance aspects.
        Example:
        - "Strong focus on renewable energy solutions."
        - "Excellent governance practices, with regular ESG reporting."

    5. **Areas for Improvement**:
        Provide actionable suggestions for addressing gaps in green finance readiness and mitigating risks highlighted by the ML model.
        Example:
        - "Increase investment in local community infrastructure."
        - "Adopt more advanced waste management practices."

    6. **Certification Recommendations**:
        Recommend certifications or frameworks the company should pursue, including timelines.
        Example:
        - "Recommended Certification: Global Reporting Initiative (GRI)."
        - "Suggested Timeline: Achieve certification within the next 12 months."

    7. **Benefits of Green Finance Alignment**:
        Explain how aligning with green finance standards will benefit the company economically and ecologically.
        Example:
        - "Enhanced access to green funding opportunities."
        - "Improved brand reputation through sustainability initiatives."

    8. **Next Steps**:
        Provide a clear roadmap for the company to improve its green finance profile and address risks.
        Example:
        - "Conduct a detailed carbon footprint analysis within the next quarter."
        - "Develop a Sustainability Action Plan addressing identified gaps."

    Use this template to create a detailed report for the user based on the input data and the recommended green finance actions.

    """
        return prompt

    def inference(self, result):
        try:
            # Prepare and send the request to the LLM model
            refined_prompt = self.prompt_template(result)
            chat_completion = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": refined_prompt,
                    }
                ],
                model=Config.GROQ_MODEL_NAME,
            )
            
            response = chat_completion.choices[0].message.content

            if response:
                logger.info("LLM inference successful.")
                return response
            else:
                logger.warning("LLM did not return any response.")
                raise APIError("LLM response is empty. Please check input format and prompt.")

        except Exception as e:
            log_exception(e, "Error during LLM inference in LLM class.")
            raise APIError("Error during LLM inference. Check input data and model configuration.") from e
