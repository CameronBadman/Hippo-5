package database

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

const (
	fileMagic   = "HIPPO5DB"
	fileVersion = uint16(1)
)

// Open loads a database from path and rebuilds its skiplist index.
func Open(path string) (*DB, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return Read(file)
}

// Save writes the database to path.
func (db *DB) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return db.Write(file)
}

// Read loads a database from r and rebuilds its skiplist index.
func Read(r io.Reader) (*DB, error) {
	magic := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != fileMagic {
		return nil, fmt.Errorf("invalid database magic")
	}

	var version uint16
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != fileVersion {
		return nil, fmt.Errorf("unsupported database version %d", version)
	}

	var dimensions int32
	if err := binary.Read(r, binary.LittleEndian, &dimensions); err != nil {
		return nil, err
	}
	if dimensions <= 0 {
		return nil, fmt.Errorf("invalid dimensions %d", dimensions)
	}

	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if count < 0 {
		return nil, fmt.Errorf("invalid record count %d", count)
	}

	db, err := New(int(dimensions))
	if err != nil {
		return nil, err
	}
	db.records = make([]Record, 0, count)

	for i := int64(0); i < count; i++ {
		record, err := readRecord(r, int(dimensions))
		if err != nil {
			return nil, fmt.Errorf("record %d: %w", i, err)
		}
		record.ID = int32(i)
		if err := db.index.Insert(record.Vector, record.ID); err != nil {
			return nil, fmt.Errorf("record %d index: %w", i, err)
		}
		db.records = append(db.records, record)
	}

	return db, nil
}

// Write serializes the database to w.
func (db *DB) Write(w io.Writer) error {
	if db == nil {
		return fmt.Errorf("nil database")
	}

	if _, err := w.Write([]byte(fileMagic)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, fileVersion); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(db.dimensions)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int64(len(db.records))); err != nil {
		return err
	}

	for i := range db.records {
		if err := writeRecord(w, db.records[i]); err != nil {
			return fmt.Errorf("record %d: %w", i, err)
		}
	}
	return nil
}

func writeRecord(w io.Writer, record Record) error {
	if err := binary.Write(w, binary.LittleEndian, int32(len(record.Vector))); err != nil {
		return err
	}
	for _, value := range record.Vector {
		if err := binary.Write(w, binary.LittleEndian, value); err != nil {
			return err
		}
	}

	if err := writeBytes(w, []byte(record.Text)); err != nil {
		return err
	}

	timestamp := record.Timestamp.UTC().UnixNano()
	if err := binary.Write(w, binary.LittleEndian, timestamp); err != nil {
		return err
	}

	metadataBytes, err := json.Marshal(record.Metadata)
	if err != nil {
		return err
	}
	return writeBytes(w, metadataBytes)
}

func readRecord(r io.Reader, dimensions int) (Record, error) {
	var vectorLen int32
	if err := binary.Read(r, binary.LittleEndian, &vectorLen); err != nil {
		return Record{}, err
	}
	if int(vectorLen) != dimensions {
		return Record{}, fmt.Errorf("dimension mismatch: expected %d, got %d", dimensions, vectorLen)
	}

	vector := make([]float32, dimensions)
	for i := range vector {
		if err := binary.Read(r, binary.LittleEndian, &vector[i]); err != nil {
			return Record{}, err
		}
	}

	textBytes, err := readBytes(r)
	if err != nil {
		return Record{}, err
	}

	var timestampNano int64
	if err := binary.Read(r, binary.LittleEndian, &timestampNano); err != nil {
		return Record{}, err
	}

	metadataBytes, err := readBytes(r)
	if err != nil {
		return Record{}, err
	}
	var metadata Metadata
	if len(metadataBytes) > 0 && string(metadataBytes) != "null" {
		metadata = make(Metadata)
		if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
			return Record{}, err
		}
	}

	return Record{
		Vector:    vector,
		Text:      string(textBytes),
		Metadata:  metadata,
		Timestamp: time.Unix(0, timestampNano).UTC(),
	}, nil
}

func writeBytes(w io.Writer, value []byte) error {
	if err := binary.Write(w, binary.LittleEndian, int64(len(value))); err != nil {
		return err
	}
	if len(value) == 0 {
		return nil
	}
	_, err := w.Write(value)
	return err
}

func readBytes(r io.Reader) ([]byte, error) {
	var length int64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	if length < 0 {
		return nil, fmt.Errorf("negative byte length %d", length)
	}
	value := make([]byte, length)
	if _, err := io.ReadFull(r, value); err != nil {
		return nil, err
	}
	return value, nil
}
