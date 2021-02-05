function get_times(start, target_length) {
    let times = [];
    let time = Date.parse(start);
    for (let j=0; j<target_length; j++) {
        times.push(time);
        time += 3600000;
    } 
    return times;
}

function get_ids(id, length) {
    let ids = [];
    for (let i=0; i<length; i++)
        ids.push(id);
    
    return ids;  
}

function get_data(db, table, id, callback) {
    let transaction = db.transaction([table]);
    let objectStore = transaction.objectStore(table);
    let request = objectStore.get(id);
    request.onerror = function(event) {
      console.log('Error '+event);
    };
    request.onsuccess = function(event) {
        if (request.result != null)
            callback(request.result);
    };  
}

function update_actuals(db, actuals, location_id) {  
    console.log("location_id="+location_id);
    console.log('updating')
    if (location_id != "") {
        get_data(db, 'actuals', location_id, function(data) {
            actuals.data.id = get_ids(data.id, data.target.length);
            actuals.data.start =  get_times(data.start, data.target.length);
            actuals.data.target = data.target; 
            actuals.data.ma = data.ma; 
            actuals.change.emit();
        });
    }    
}
  
function update_predictions(db, predictions, location_id, prediction_id) {        
    if (location_id != "") {
        let idx = location_id + ":" + prediction_id;
        get_data(db, 'predictions', idx, function(data) {
            var length = null;
            if (predictions.data.start != null)
                length = predictions.data.start.length
            
            for (let q=1; q<10; q++) {
                let quantile = '0.'+q;
                if (quantile in data) {
                    predictions.data[quantile] = data[quantile];
                    length = data[quantile].length;
                }
            }   
            if (length != null) {
                predictions.data.start = get_times(data.start,length);
                predictions.data.id = get_ids(data.id, length);
                predictions.change.emit();
            } 
        });
    }
}

let location_id = "" + parseInt(location_select.value); 
let prediction_id = "" + parseInt(start_slider.value);
let open_request = indexedDB.open("openaq", 1);
open_request.onsuccess = function(event) {
    let db = open_request.result;
    update_actuals(db, actuals, location_id);
    update_predictions(db, predictions, location_id, prediction_id);
};

