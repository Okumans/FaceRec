import shelve

# with shelve.open('angle_result_result') as db:
#     for i in db["result"]:
#         score = 0
#         max_score = 0
#         del db["result"][i]["24_d8919"]
#         del db["result"][i]["14_cef21"]
#         del db["result"][i]["3_02df7"]
#
#         for j in db["result"][i]:
#             score += sum(db["result"][i][j])
#             max_score += len(db["result"][i][j])
#             # print(j, sum(db["result"][i][j])/len(db["result"][i][j])*100)
#         print(f"total score {i} is {score/max_score*100}")

with shelve.open("distance_result_result_1") as db:
    for i in db["result"]:
        score = 0
        max_score = 0
        del db["result"][i]["24_d8919"]
        del db["result"][i]["14_cef21"]
        del db["result"][i]["3_02df7"]

        for j in db["result"][i]:
            score += sum(db["result"][i][j])
            max_score += len(db["result"][i][j])
            # print(j, sum(db["result"][i][j])/len(db["result"][i][j])*100)
        print(f"total score {i} is {score/max_score*100}")
