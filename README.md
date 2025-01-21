# LLM_Distribution_Center
code implementation of center
1. establish the connection between edge node and center, register all the edge node and keep the connection alive for the whole process
2. get input data from local enviroment and generate input for each round
3. 20 task offloading rounds before model offloading round
4. after algorithem finished, match the target edge id with task offloading decisions and model caching decisions
5. call the transmission function and distribute the specific tasks to edge node
6. call the function for model transmission
7. after model deployment ready, ready for next round


## sample_data data structure  maintaniner: @wxhfj
* task_id : 
    the primary key 
* task_type: 
    * **0 TC (Text Classification)**
        * reference_enum:
            * 0 : sad
            * 1 : happy
            * 2 : love
            * 3 : angry
            * 4 : scared
            * 5 : surprise
    * **1 NER (Named Entity Recognition)**
        * reference_enum:
            * O: None 
            * B-\<ENTITY\> Beginning part of certain words
            * I-\<ENTITY\> Internal part of certain words
        ENTITY : PER(Person Name) LOC(Location) ORG(Organization)  MISC(Miscellaneous)
    * **2 QA (Question Answering)**
        * reference_value: ideal standard answer
    * **3 TS (Translation Chinese to English)** 
        * reference_value: ideal standard answer
    * **4 SG (Summarization Generation)**
        * reference_value: ideal standard answer
* task_token : 
    query words

