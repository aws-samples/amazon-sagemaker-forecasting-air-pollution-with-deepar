function update_location(location_select, map_source) {
    let options = [];
    for (let i of map_source.selected.indices) {
        let desc = map_source.data.id[i] + ' - ' +
                   map_source.data.country[i] + ' - ' +
                   map_source.data.city[i] + ' - ' + 
                   map_source.data.location[i];
        options.push(desc);
    }
    
    location_select.options = options;
    location_select.value = null;   
    location_select.change.emit();   
}

update_location(location_select, map_source);