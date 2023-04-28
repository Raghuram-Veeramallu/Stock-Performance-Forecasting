CREATE TABLE IF NOT EXISTS {table_name} (
	id INT PRIMARY KEY,
	company TEXT NOT NULL,
	"year" INT NOT NULL,
	quarter INT NOT NULL,
	"date" TEXT NOT NULL,
	speaker TEXT,
	actual_transcript TEXT,
	summarized_transcript TEXT,
	label TEXT,
	score DOUBLE
);
