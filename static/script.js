document.addEventListener("DOMContentLoaded", function() {
    var map = L.map('map').setView([37.7749, -122.4194], 5);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    fetch('/get_locations')
        .then(response => response.json())
        .then(data => {
            data.forEach(location => {
                L.marker([location.lat, location.lng])
                    .addTo(map)
                    .bindPopup(`<b>${location.animal} detected here!</b>`);
            });
        });
});
