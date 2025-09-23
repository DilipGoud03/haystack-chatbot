from pipelines import search_query

if __name__ == '__main__':
    while True:
        print('--'*50)
        question = input('Enter your query (or exit to quit) :')
        print('--'*50, '\n')
        if question.lower() == 'exit':
            break
        result = search_query(question)
        print('\n', '--'*50)
        print('Answer :', result["llm"]["replies"][0].text)
        print('--'*50, '\n')
