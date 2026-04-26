package database_test

import (
	"fmt"

	"hippo5/database"
)

func ExampleDB_Search() {
	db, _ := database.New(3)
	db.Insert([]float32{0.1, 0.2, 0.3}, "dark mode preference", database.Metadata{"agent": "alice"})
	db.Insert([]float32{0.8, 0.1, 0.1}, "unrelated memory", database.Metadata{"agent": "alice"})

	results, _ := db.Search([]float32{0.1, 0.2, 0.3}, database.SearchOptions{
		Epsilon:   0.05,
		Threshold: 0,
		TopK:      1,
		Filter: &database.Filter{
			Metadata: map[string]any{"agent": "alice"},
		},
	})

	fmt.Println(results[0].Record.Text)
	// Output: dark mode preference
}
