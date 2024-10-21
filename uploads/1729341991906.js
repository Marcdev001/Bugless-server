
document.getElementById('searchInput').addEventListener('input', debounce(handleSearch, 300));

function debounce(func, delay) {
    let timeoutId;
    return function () {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(func, delay);
    };
}

async function handleSearch() {
    const searchTerm = document.getElementById('searchInput').value;

    if (searchTerm.trim() !== '') {
        const data = await fetchData(searchTerm);
        displayResults(data);
    } else {
        clearResults();
    }
}

async function fetchData(searchTerm) {
    const spaceId = 'aoq6ts9450dx';
    const accessToken = 'tYfnXw-LA5GhFsSRnmjJDQiXg-XKJqtyiCEp1tC9hsQ';
    const apiUrl = `https://cdn.contentful.com/spaces/${spaceId}/entries?access_token=${accessToken}&content_type=product&fields.name[match]=${searchTerm}`;

    try {
        const response = await fetch(apiUrl);
        const data = await response.json();
        return data.items;
    } catch (error) {
        console.error('Error fetching data:', error);
        return [];
    }
}

function displayResults(data) {
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = '';

    data.forEach(product => {
        const productName = product.fields.name;
        const price = product.fields.price;
        const details = product.fields.details;

        const resultItem = document.createElement('div');
        resultItem.innerHTML = `<strong>${productName}</strong><br>Price: ${price}<br>Details: ${details}<hr>`;
        searchResults.appendChild(resultItem);
    });
}

function clearResults() {
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = '';
}
