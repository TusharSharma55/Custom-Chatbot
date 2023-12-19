<?php
/**
* Plugin Name: FastAPI Integration
* Description: Integration of FastAPI with WordPress.
* Version: 1.0
* Author: Your Name
*/
 
// Activation hook
register_activation_hook(__FILE__, 'fastapi_integration_activate');
 
function fastapi_integration_activate() {
    // Additional activation logic, if needed
}
 
// Deactivation hook
register_deactivation_hook(__FILE__, 'fastapi_integration_deactivate');
 
function fastapi_integration_deactivate() {
    // Additional deactivation logic, if needed
}
 
// Define the endpoint URL of your FastAPI application
// $fastapi_endpoint = 'https://custom-chatbot3.onrender.com';
$fastapi_endpoint = 'https://5cf7-115-245-112-218.ngrok-free.app'; 

 // Enqueue styles and scripts
function fastapi_integration_enqueue_styles_and_scripts() 
{
    wp_enqueue_style('fastapi-integration-styles', plugin_dir_url(__FILE__) . 'assests/style.css');
    wp_enqueue_script('fastapi-integration-script', plugin_dir_url(__FILE__) . 'assests/script.js', array('jquery'), null, true);
}
add_action('wp_enqueue_scripts', 'fastapi_integration_enqueue_styles_and_scripts');


// Add a WordPress REST API endpoint`
add_action('rest_api_init', function () use ($fastapi_endpoint) {
    register_rest_route('fastapi_integration/v1', '/user-input/', array(
        'methods' => 'POST',
        'callback' => 'fastapi_integration_get_answer',
    ));
});

// Function to handle the API request and communicate with FastAPI
function fastapi_integration_get_answer() {
    global $fastapi_endpoint;  // Made $fastapi_endpoint global
 
    $response = wp_remote_get($fastapi_endpoint);
 
    if (is_wp_error($response)) {
        return array('error' => 'Error communicating with FastAPI.');
    } else {
        $body = wp_remote_retrieve_body($response);
        $data = json_decode($body, true);
        return $data;
    }
}

// Add the chatbot container to the footer
function fastapi_integration_add_chatbot_container() {
    ?>
<div id="chatbot-container"></div>
<?php
}
 
add_action('wp_footer', 'fastapi_integration_add_chatbot_container');
?>