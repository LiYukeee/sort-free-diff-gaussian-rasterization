#include "rasterize_points.h"
#include <torch/script.h>
#include <memory>
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include <torch/torch.h>
#include <filesystem>

void printTensorInfo(const torch::Tensor &tensor, const bool print_data = false)
{
    std::cout << "Tensor size, type, device: " << tensor.sizes() << " " << tensor.dtype() << " " << tensor.device() << std::endl;
    if (print_data)
    {
        std::cout << "Tensor data: " << std::endl
                  << tensor << std::endl;
    }
}

std::vector<char> get_the_bytes(std::string filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}

torch::Tensor load_tensor(std::string filename)
{
    std::vector<char> f = get_the_bytes(filename);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor my_tensor = x.toTensor();
    return my_tensor;
}

template <typename T>
T load_scalar(std::string filename)
{
    std::vector<char> f = get_the_bytes(filename);
    torch::IValue x = torch::pickle_load(f);
    T my_scalar = x.toScalar().to<T>();
    return my_scalar;
}

void runForward(const std::filesystem::path args_path)
{
    // sort free variable
    const float sigma = load_scalar<float>(args_path / "sigma.pt");
    const float weight_background = load_scalar<float>(args_path / "weight_background.pt");
    const torch::Tensor vi = load_tensor(args_path / "vi.pt").to(torch::kCUDA);

    // // anchor parameters
    // const torch::Tensor anchors3D = load_tensor(args_path / "anchors3D.pt").to(torch::kCUDA);
    // const torch::Tensor anchor_scales = load_tensor(args_path / "anchor_scales.pt").to(torch::kCUDA);

    const torch::Tensor background = load_tensor(args_path / "bg.pt").to(torch::kCUDA);
    const torch::Tensor means3D = load_tensor(args_path / "means3D.pt").to(torch::kCUDA);
    const torch::Tensor colors = load_tensor(args_path / "colors_precomp.pt").to(torch::kCUDA);
    const torch::Tensor opacity = load_tensor(args_path / "opacities.pt").to(torch::kCUDA);
    const torch::Tensor scales = load_tensor(args_path / "scales.pt").to(torch::kCUDA);
    const torch::Tensor rotations = load_tensor(args_path / "rotations.pt").to(torch::kCUDA);
    const float scale_modifier = load_scalar<float>(args_path / "scale_modifier.pt");
    const torch::Tensor cov3D_precomp = load_tensor(args_path / "cov3Ds_precomp.pt").to(torch::kCUDA);
    const torch::Tensor viewmatrix = load_tensor(args_path / "viewmatrix.pt").to(torch::kCUDA);
    const torch::Tensor projmatrix = load_tensor(args_path / "projmatrix.pt").to(torch::kCUDA);
    const float tanfovx = load_scalar<float>(args_path / "tanfovx.pt");
    const float tanfovy = load_scalar<float>(args_path / "tanfovy.pt");

    const int image_height = load_scalar<int>(args_path / "image_height.pt");
    const int image_width = load_scalar<float>(args_path / "image_width.pt");
    const torch::Tensor sh = load_tensor(args_path / "sh.pt").to(torch::kCUDA);
    const int sh_degree = load_scalar<int>(args_path / "sh_degree.pt");
    const torch::Tensor campos = load_tensor(args_path / "campos.pt").to(torch::kCUDA);
    const bool prefiltered = load_scalar<bool>(args_path / "prefiltered.pt");
    const bool debug = load_scalar<bool>(args_path / "debug.pt");
    const bool if_depth_correct = load_scalar<bool>(args_path / "if_depth_correct.pt");


    int rendered_gs;
    torch::Tensor out_color, out_depth, accum_weights_ptr, accum_weights_count, accum_max_count, radii, geomBuffer, binningBuffer, imgBuffer;
    std::tie(rendered_gs, out_color, radii, geomBuffer, binningBuffer, imgBuffer) = RasterizeGaussiansCUDA(
        sigma,
        weight_background,
        background,
        means3D,
        colors,
        opacity,
        vi,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
        sh,
        sh_degree,
        campos,
        prefiltered,
        debug,
        if_depth_correct
    );
    std::string filename_color = "../output/render.pt";
    try
    {
        torch::save(out_color, filename_color);
        std::cout << "Tensor saved successfully to " << filename_color << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error saving tensor to " << filename_color << ": " << e.what() << std::endl;
    }
}


// void runBackward(const std::filesystem::path args_path)
// {
//     ///////////////BACKWARD///////////////
//     std::cout << "Running Backward: " << std::endl;

//     const float sigma = load_scalar<float>(args_path / "sigma.pt");;
//     const torch::Tensor opacity_weight = load_tensor(args_path / "opacity_weight.pt").to(torch::kCUDA);
//     const torch::Tensor opacities = load_tensor(args_path / "opacities.pt").to(torch::kCUDA);
//     const torch::Tensor out_color = load_tensor(args_path / "render_image.pt").to(torch::kCUDA);
//     const torch::Tensor error_map = load_tensor(args_path / "error_map.pt").to(torch::kCUDA);
//     const float scene_radii = load_scalar<float>(args_path / "scene_radii.pt");
//     const torch::Tensor sh = load_tensor(args_path / "sh.pt").to(torch::kCUDA);
//     const int sh_degree = load_scalar<int>(args_path / "sh_degree.pt");
//     const torch::Tensor radii = load_tensor(args_path / "radii.pt").to(torch::kCUDA);

//     const torch::Tensor background = load_tensor(args_path / "bg.pt").to(torch::kCUDA);
//     const torch::Tensor means3D = load_tensor(args_path / "means3D.pt").to(torch::kCUDA);
//     const torch::Tensor colors = load_tensor(args_path / "colors_precomp.pt").to(torch::kCUDA);
//     const torch::Tensor scales = load_tensor(args_path / "scales.pt").to(torch::kCUDA);
//     const torch::Tensor rotations = load_tensor(args_path / "rotations.pt").to(torch::kCUDA);
//     const float scale_modifier = load_scalar<float>(args_path / "scale_modifier.pt");
//     const torch::Tensor cov3D_precomp = load_tensor(args_path / "cov3Ds_precomp.pt").to(torch::kCUDA);
//     const torch::Tensor viewmatrix = load_tensor(args_path / "viewmatrix.pt").to(torch::kCUDA);
//     const torch::Tensor projmatrix = load_tensor(args_path / "projmatrix.pt").to(torch::kCUDA);
//     const float tanfovx = load_scalar<float>(args_path / "tanfovx.pt");
//     const float tanfovy = load_scalar<float>(args_path / "tanfovy.pt");
//     const torch::Tensor grad_color = load_tensor(args_path / "grad_out_color.pt").to(torch::kCUDA);
//     const torch::Tensor campos = load_tensor(args_path / "campos.pt").to(torch::kCUDA);
//     const torch::Tensor geomBuffer = load_tensor(args_path / "geomBuffer.pt").to(torch::kCUDA);
//     const torch::Tensor binningBuffer = load_tensor(args_path / "binningBuffer.pt").to(torch::kCUDA);
//     const torch::Tensor imgBuffer = load_tensor(args_path / "imgBuffer.pt").to(torch::kCUDA);
//     const bool debug = load_scalar<bool>(args_path / "debug.pt");
//     const int rendered_gs = load_scalar<int>(args_path / "num_rendered.pt");
//     // printTensorInfo(viewmatrix, true);
//     torch::Tensor dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dopacity_weight, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dsigma, dL_dweight_background;
//     std::tie(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dopacity_weight, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dsigma, dL_dweight_background) = RasterizeGaussiansBackwardCUDA(
//         sigma,
//         out_color,
//         opacity_weight,
//         opacities,
//         background,
//         means3D,
//         radii,
//         colors,
//         scales,
//         rotations,
//         scene_radii,
//         scale_modifier,
//         cov3D_precomp,
//         viewmatrix,
//         projmatrix,
//         tanfovx,
//         tanfovy,
//         grad_color,
//         sh,
//         sh_degree,
//         campos,
//         geomBuffer,
//         rendered_gs,
//         binningBuffer,
//         imgBuffer,
//         debug
//         );
//     torch::save(dL_dmeans2D, "dL_dmeans2D");
//     torch::save(dL_dcolors, "dL_dcolors");
//     torch::save(dL_dopacity, "dL_dopacity");
//     torch::save(dL_dopacity_weight, "dL_dopacity_weight");
//     torch::save(dL_dmeans3D, "dL_dmeans3D");
//     torch::save(dL_dcov3D, "dL_dcov3D");
//     torch::save(dL_dsh, "dL_dsh");
//     torch::save(dL_dscales, "dL_dscales");
//     torch::save(dL_drotations, "dL_drotations");
//     torch::save(dL_dsigma, "dL_dsigma");
//     torch::save(dL_dweight_background, "dL_dweight_background");
// }



int main()
{
    std::filesystem::path args_path = "../test_data";
    runForward(args_path / "forward_tensors");
    // runBackward(args_path / "backward_tensors");
    return 0;
}