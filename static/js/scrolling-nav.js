

//jQuery to collapse the navbar on scroll
$(window).scroll(function() {
    if ($(".navbar").offset().top > 50) {
        $(".navbar-fixed-top").addClass("top-nav-collapse");
    } else {
        $(".navbar-fixed-top").removeClass("top-nav-collapse");
    }
});

//jQuery for page scrolling feature - requires jQuery Easing plugin
$(function() {
    $(document).on('click', 'a.page-scroll', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top
        }, 1500, 'easeInOutExpo');
        event.preventDefault();
    });
});


// Code to handle send email

$('#send_email').click(function(event){
    event.preventDefault();
    var data = {
        displayName: $('#displayName').val(),
        emailAddress: $('#emailAddress').val(),
        message: $('#message').val()
    };
    $.ajax({
        url: '/send_email',
        dataType: 'json',
        type: 'post',
        contentType: 'application/json',
        data: JSON.stringify(data),
        processData: false,
        success: function( data, textStatus, jQxhr ){
            console.log(data);
            document.getElementById("contact").reset();
            $('#myModal').modal('show');

            $('#json-renderer').jsonViewer(data);

            $('#myModal').on('hidden.bs.modal', function (e) {
                $('#json-renderer').jsonViewer({});
            });
        },
        error: function( jqXhr, textStatus, errorThrown ){
            console.log( errorThrown );
            $('#json-renderer').jsonViewer(errorThrown);
        }
    });
});