with open('text_file.txt', 'r', encoding='utf8') as f:
	print(f.tell())
	line = f.next()
	print(f.tell())