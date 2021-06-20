package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

type Distance struct {
	Index int
	Value float64
}

type Knn struct {
	K int
	X [][]float64
	Y []int
}

func NewKnn(k int) *Knn {
	return &Knn{
		K: k,
	}
}

/**
Tinh khoang cach tu 2 diem dua tren cac features dung cong thuc euclidean
p1 testing
p2 training
*/
func (knn *Knn) distance(p1 []float64, p2 []float64) float64 {
	var value float64 = 0
	for i := 0; i < len(p1); i++ {
		value += math.Pow(p1[i]-p2[i], 2)
	}
	return math.Sqrt(value)
}

func (knn *Knn) Fit(x [][]float64, y []int) {
	knn.X = make([][]float64, len(x))
	for i := range x {
		knn.X[i] = make([]float64, len(x[i]))
		copy(knn.X[i], x[i])
	}
	knn.Y = make([]int, len(y))
	copy(knn.Y, y)
}

func (knn *Knn) Train(data []float64, label int) {
	knn.X = append(knn.X, data)
	knn.Y = append(knn.Y, label)
}

// Predict
//Du doan nhan(labels) tu du lieu
//@param x  mang 2 chieu chua cac features , cho phep du doan 1 luc nhieu mau
//@return vector 1 chieu labels ung voi tap data dau vao /**
func (knn *Knn) Predict(data [][]float64) []int {
	var (
		results   []int
		distances []*Distance
	)
	for i := 0; i < len(data); i++ {
		// tinh khoang cach
		for j := 0; j < len(knn.X); j++ {
			distances = append(distances, &Distance{
				Index: j,
				Value: knn.distance(data[i], knn.X[j]),
			})
		}
		// sort lai khoang cach tu nho toi to
		sort.SliceStable(distances, func(i, j int) bool {
			return distances[i].Value < distances[j].Value
		})
		// chon ra k phan tu va dem
		counter := make(map[int]int)
		for j := 0; j < len(knn.Y); j++ {
			counter[knn.Y[j]] = 0
		}
		// tien hanh dem
		for j := 0; j < knn.K; j++ {
			labelValue := knn.Y[distances[j].Index]
			counter[labelValue] = counter[labelValue] + 1
		}
		// chon ra nhan co so dem cao nhat
		var (
			maxValue = 0
			maxIndex = 0
		)
		for k, v := range counter {
			println("weight:", k, v)
		}
		for j := 0; j < len(knn.Y); j++ {
			labelValue := knn.Y[j]
			if counter[labelValue] > maxValue {
				maxIndex = j
				maxValue = counter[labelValue]
			}
		}
		results = append(results, knn.Y[maxIndex])
	}

	return results
}

func printData(x [][]float64) {
	//println("debug", len(x), len(x[0]))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[i]); j++ {
			if j > 0 {
				print(" ")
			}
			print(fmt.Sprintf("%v", x[i][j]))
		}
		print("\n")
	}
}

func increaseItemData(value float64, row, col int) float64 {
	return value * 10 * float64(row+col)
}

func readTest(filename string) []float64 {
	data, err := ioutil.ReadFile(fmt.Sprintf("./testing/%s", filename))
	if err != nil {
		panic(err)
	}
	s := string(data)
	lines := strings.Split(s, "\n")
	var (
		r [][]float64
	)
	for i := 0; i < len(lines); i++ {
		var row []float64
		for j := 0; j < len(lines[i]); j++ {
			if string(lines[i][j]) != " " {
				v, _ := strconv.ParseFloat(string(lines[i][j]), 64)
				row = append(row, v)
			}
		}
		r = append(r, row)
	}
	if len(r) == 0 {
		panic("empty testing")
	}
	var response []float64
	for i := 0; i < len(r); i++ {
		for j := 0; j < len(r[0]); j++ {
			response = append(response, increaseItemData(r[i][j], i, j))
		}
	}
	return response
}

func readDataSet(filename string) [][]float64 {
	data, err := ioutil.ReadFile(fmt.Sprintf("./dataset/%s", filename))
	if err != nil {
		panic(err)
	}
	s := string(data)
	lines := strings.Split(s, "\n")
	// duyet theo cot dataset
	var (
		results [][][]float64 // luu mang cac result
		result  [][]float64
		counter = 0
	)
	for i := 0; i < len(lines)-1; i++ { // bo 1 mau cuoi ra de test , thuc ra tot nhat la random
		var row []float64
		for j := 0; j < len(lines[i])-1; j++ {
			if string(lines[i][j]) != " " {
				v, _ := strconv.ParseFloat(string(lines[i][j]), 64)
				row = append(row, v)
			}
		}
		result = append(result, row)
		counter++
		if counter == 24 {
			counter = 0
			results = append(results, result)
			result = [][]float64{} // reset result
			continue
		}

		row = []float64{} // reset array hang
	}
	/*for i := 0; i < len(results); i++ {
		printData(results[i])
		println("--------------------------------")
	}*/

	// tinh toan va dua ve mang 1 chieu cua tung result
	var response [][]float64
	for _, r := range results {
		var res []float64
		for i := 0; i < len(r); i++ {
			// trong so nhat voi hang + trong so * voi cot
			//var value float64 = 0
			for j := 0; j < len(r[0]); j++ {
				//value += r[i][j] * float64(i + j)
				res = append(res, r[i][j]*10*float64(i+j))
			}
			//res = append(res, value)
		}
		response = append(response, res)
	}
	return response //[]float64{}
}

func test() {
	/*knn := NewKnn(9)
	knn.Fit([][]float64{
		{0.5},
		{3.0},
		{4.5},
		{4.6},
		{4.9},
		{5.2},
		{5.3},
		{5.5},
		{7.0},
		{9.5},
	}, []int{
		0,
		0,
		1,
		1,
		1,
		0,
		0,
		1,
		0,
		0,
	})
	r := knn.Predict([][]float64{
		{
			 5.0,
		},
	})
	println(r[0])*/
}

var (
	knn = NewKnn(11)
)

type ResponsePayload struct {
	Value int `json:"value"`
	Error *string
}

type TrainPayload struct {
	Data  [][]float64 `json:"data"`
	Label int         `json:"label"`
}

func httpServer() {
	processData := func(d [][]float64) []float64 {
		var res []float64
		for i := 0; i < len(d); i++ {
			for j := 0; j < len(d[0]); j++ {
				res = append(res, increaseItemData(d[i][j], i, j))
			}
		}

		return res
	}

	fmt.Println("server started: http://127.0.0.1:8080")
	http.Handle("/", http.FileServer(http.Dir("./public")))
	http.HandleFunc("/api/predict", func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		var testingData [][]float64
		err := decoder.Decode(&testingData)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		printData(testingData)
		data := processData(testingData)
		result := knn.Predict([][]float64{data})
		w.Header().Set("Content-Type", "application/json")
		payload := &ResponsePayload{
			Value: result[0],
			Error: nil,
		}
		json.NewEncoder(w).Encode(payload)
	})
	http.HandleFunc("/api/train", func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		var jsonData TrainPayload
		err := decoder.Decode(&jsonData)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		data := processData(jsonData.Data)
		knn.Train(data, jsonData.Label)
		w.Header().Set("Content-Type", "application/json")
		payload := &ResponsePayload{
			Value: jsonData.Label,
			Error: nil,
		}
		json.NewEncoder(w).Encode(payload)
	})
	http.ListenAndServe(":8080", nil)
}

func main() {
	var (
		x [][]float64
		y []int
	)
	for label := 0; label <= 9; label++ {
		data := readDataSet(fmt.Sprintf("class%d.txt", label))
		for i := 0; i < len(data); i++ {
			x = append(x, data[i])
			y = append(y, label)
		}
	}
	knn.Fit(x, y) // bat dau training model
	// du doan nhan
	testClass := readTest("data.txt")
	testData := [][]float64{
		testClass,
	}
	result := knn.Predict(testData)
	println("result", result[0])

	httpServer()
}
