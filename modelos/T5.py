
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")



list = [ "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: The new Inflation Reduction Act has just been updated to give Americans making less than $50,000 per year up to $6,400 in subsidies every single month.   Label: false   Summary Explanation:The U.S. government has not updated the Inflation Reduction Act, a 2022 law, to give Americans $6,400 in monthly subsidies.The link included with the post is not affiliated with the federal government.",
         "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: Credit card debt is above $1 trillion for the FIRST TIME EVER   Label: mostly-true  Summary Explanation: Numerically, West Virginia Gov. Jim Justice is right that national credit card debt topped $1 trillion for the first time, according to data from the New York Federal Reserve.Focusing on credit card debt’s absolute value ignores that consumer debt is shrinking compared with the broader U.S. economy. Focusing on this number also leaves out that credit card debt accounts for only a modest proportion of all consumer debt.    ",
          "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: Russia has hypersonic missile capabilities ahead of that of the U.S.   Label: half-true  Summary Explanation: Russia has deployed hypersonic missiles — those that travel five times the speed of sound — in its war against Ukraine, while the U.S. is still developing hypersonic missiles.Russia’s hypersonic missiles are considered primitive. The U.S. versions are expected to have sophisticated abilities to glide, be launched by aircraft and maneuver to avoid defenses. Experts say they are a couple of years from being deployed.    ",
           "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: You’re in danger because the American Red Cross doesn’t label blood donations from donors vaccinated against COVID-19.   Label: false  Summary Explanation: There is no danger from blood transfusions that include blood from COVID-19 vaccinated donors, health experts said.Vaccine components are not found in the bloodstream, an American Red Cross spokesperson said.   ",
            "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: 2022 was the biggest tourism year ever in West Virginia.    Label: half-true  Summary Explanation: Spending by travelers to West Virginia set a record in 2022.However, tourism-supported jobs fell to a level unseen since 2007, and tax revenue from tourism fell slightly from 2021 levels.    ",
             "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: NASA admits space and planets are fake.  Label: pants-fire  Summary Explanation: NASA didn’t say space and planets are fake.   ",
              "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: U.S sell arms to 60% of the world's autocrats. They are the world's largest arms exporters.  Label: true  Summary Explanation: The U.S. is the world’s largest arms exporter, with 40% of the global share, easily ahead of Russia at 16%.   ",
               "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: 1936 in the United States was much hotter than 2023.   Label: false  Summary Explanation: National Oceanic and Atmospheric Administration data shows that in the contiguous U.S., the first nine months of 2023 were 1.64°F warmer than the parallel period in 1936.There’s overwhelming evidence that human-caused climate change is warming the Earth and leading to more intense and frequent heat waves.   ",
                "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: Elon Musk is dead.   Label: false  Summary Explanation: This claim is unfounded. X owner Elon Musk is alive.   ",
                 "Generate: Write me a joke that exlains why the claim has that label taking into account the summary     Claim: New York Attorney General Letitia James said President Joe Biden is an embarrassment to all that we stand for.  Label: false  Summary Explanation: Video footage of New York Attorney General Letitia James shows her criticizing then-President Donald Trump in 2018, not President Joe Biden now.   "]



promt_time = "Write me a joke that exlains why the claim has that label taking into account the summary     Claim: Credit card debt is above $1 trillion for the FIRST TIME EVER   Label: mostly-true  Summary Explanation: Numerically, West Virginia Gov. Jim Justice is right that national credit card debt topped $1 trillion for the first time, according to data from the New York Federal Reserve.Focusing on credit card debt’s absolute value ignores that consumer debt is shrinking compared with the broader U.S. economy. Focusing on this number also leaves out that credit card debt accounts for only a modest proportion of all consumer debt.   Joke: "

start_time = time.time()


input_ids = tokenizer.encode(promt_time, return_tensors="pt", truncation=True) 
greedy_output = model.generate(input_ids, num_beams=5, no_repeat_ngram_size=2, min_length=50, max_length=100)

generated_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

print(generated_text)

end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")