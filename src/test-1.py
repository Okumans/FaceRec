raw_split_name = "jeerabhat a".split()

firstname = raw_split_name[0] if raw_split_name else ""
lastname = " ".join(raw_split_name[1:]) if len(raw_split_name) > 1 else ""
print(firstname, ":",  lastname)