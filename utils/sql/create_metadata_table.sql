CREATE TABLE IF NOT EXISTS predictions_metadata (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	table_name TEXT NOT NULL,
	method_name TEXT NOT NULL,
	author TEXT NOT NULL,
	summarization_model TEXT NOT NULL, 
	classification_model TEXT NOT NULL, 
	"date" TEXT NOT NULL
);
